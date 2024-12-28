#!/usr/bin/env python
import torch
from torch import nn 
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import torch.nn.functional as F
import torch.backends.cudnn as cudnn #ChatGPT: optimize the performance of convolutional neural network computations on NVIDIA GPUs

import copy
import numpy as np
from numpy import linalg as LA
import random
from tqdm import trange
import h5py
import random
import itertools
import collections
import sys

from utils.distribute import uniform_distribute, train_dg_split
from utils.sampling import iid, noniid_cifar
import argparse

from src.nets import MLP, CNN_v1, CNN_v2, CNN_v3, Alexnet, get_model
from src.strategy import FedAvg
# from src.test import test_img
from src.attack import visualize_adversarial_images
from src.resnet import ResNet18

import pickle
import os

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
# plt.style.use('seaborn')
# plt.rcParams.update({'lines.markeredgewidth': 1})

from datetime import datetime

import math


# # class Args - Change 3

# In[8]:

def args_parser():
    parser = argparse.ArgumentParser()
    
    # federated arguments
    parser.add_argument('--fed', type=str, default='fedavg', help="federated optimization algorithm")
    parser.add_argument('--mu', type=float, default=1e-2, help='hyper parameter for fedprox')
    parser.add_argument('--rounds', type=int, default=100, help="total number of communication rounds")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="number of local epochs: E")
    parser.add_argument('--min_le', type=int, default=5, help="minimum number of local epoch")
    parser.add_argument('--max_le', type=int, default=15, help="maximum number of minimum local epoch")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=100, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="client learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--l2_lambda', type=float, default=0.0002, help="SGD l2 regularization (default: 0.992)")
    parser.add_argument('--classwise', type=int, default=1000, help="number of images for each class (global dataset)")
    parser.add_argument('--alpha', type=float, default=0.5, help="random portion of global dataset")
    
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--model', type=str, default='customized_resnet18', help='model name')
    parser.add_argument('--sampling', type=str, default='noniid', help="sampling method")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=123, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--sys_homo', action='store_true', help='no system heterogeneity')
    parser.add_argument('--tsboard', action='store_true', help='tensorboard')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--filepath', type=str, help="Model file path")
    #Stage-data sharing
    parser.add_argument('--stage_1_round', type=int, default=100, help="time to offer data sharing")
    parser.add_argument('--global_lr', type=int, default=1, help="SCAFFOLD param")

    #softlabel-AT
    parser.add_argument('--soft_label_clean', type=float, default=0.95, help="soft label value for the true class of the clean train examples")
    parser.add_argument('--mean', type=float, default=0, help="gaussian noise mean")
    parser.add_argument('--sigma', type=float, default=0.1, help="gaussian noise std")
    parser.add_argument('--rho', type=float, default=0.5, help="early stoppping criteria - perturbation limit")

    #PGD
    parser.add_argument('--eps', type=float, default=0.0314, help="PGD eps")
    parser.add_argument('--nb_iter', type=int, default=7, help="PGD nb_iter")
    parser.add_argument('--eps_iter', type=float, default=0.00784, help="PGD step size")
    parser.add_argument('--clip_min', type=float, default=0.0, help="PGD neighbor lower limit")
    parser.add_argument('--clip_max', type=float, default=1.0, help="PGD neighbor upper limit")

    #FGSM
    parser.add_argument('--eps_FGSM', type=float, default=0.031, help="FGSM perturbation limit")
    parser.add_argument('--pretrained', action='store_true', help="pre-trained")
    args = parser.parse_args()

    return args

start_time = datetime.now() 

args = args_parser()

args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
# # Reproducibility

# In[10]:


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)     
torch.cuda.manual_seed_all(args.seed)


# # Load dataset and split users - Change 1

# In[11]:


def train_dg_split(dataset, args): 
    dg_idx = []
    train_idx = []
    idxs = np.arange(len(dataset))

    if args.dataset == "mnist":
        labels = dataset.targets.numpy()
    elif args.dataset == "cifar" or args.dataset == "adv_cifar":
        labels = np.array(dataset.targets)
    else:
        exit('Error: unrecognized dataset')
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    
    idxs = idxs_labels[0]
    labels = idxs_labels[1]
    
    #for all classes
    for i in range(args.num_classes):
        specific_class = np.extract(labels == i, idxs)
        
        # uniformly assign the particular class (args.classwise) to the global dataset
        dg = np.random.choice(specific_class, args.classwise, replace=False)
        
        # divide and split the datata into the global dataset and all parties' dataset
        train_tmp = set(specific_class)-set(dg)
        
        dg_idx = dg_idx + list(dg)
        
        train_idx = train_idx + list(train_tmp)
    
    return dg_idx, train_idx 


# In[12]:


def noniid(dataset, args):
    """
    Sample non-I.I.D client data from dataset
    -> Different clients can hold vastly different amounts of data
    :param dataset:
    :param num_users:
    :return:
    """
    num_dataset = len(dataset)
    idx = np.arange(num_dataset)
    dict_users = {i: list() for i in range(args.num_users)}
    
    min_num = int(num_dataset/args.num_users)
    max_num = int(num_dataset/args.num_users)

    random_num_size = np.random.randint(min_num, max_num+1, size=args.num_users)
    print(f"Total number of datasets owned by clients : {sum(random_num_size)}")

    # total dataset should be larger or equal to sum of splitted dataset.
    assert num_dataset >= sum(random_num_size)

    # divide and assign
    for i, rand_num in enumerate(random_num_size):

        rand_set = set(np.random.choice(idx, rand_num, replace=False))
        idx = list(set(idx) - rand_set)
        dict_users[i] = rand_set

    return dict_users

def print_statistics(i, y_train_pi, nb_labels=10):
    print('Party_', i)
    for l in range(nb_labels):
        print('* Label ', l, ' samples: ', (y_train_pi == l).sum())

#Source: https://colab.research.google.com/drive/1YoFBvGAdBSlLZwS-R6kmOWD7eX9cAljt#scrollTo=mH2gC7c9aJrn
def count_unique_values(client_label_map, num_classes):
    value_counts = collections.Counter()
    for value_list in client_label_map.values():
      value_counts.update(value_list)

    # Print the frequency of each unique value
    for value in range(num_classes):
        count = value_counts[value]
        print(f"{value}: {count}")

def generate_label_map(seed=123, n_combinations=2, num_classes=10):
    # Set the random seed for reproducibility
    random.seed(seed)

    # Generate all possible combinations of 2 integers from 0 to 9
    # Generate a list of all possible combinations of 2 elements from a list of 10 elements
    elements = list(range(num_classes))
    combinations = list(itertools.combinations(elements, 2))

    # Randomly select n_combinations of the possible combinations
    chosen_combinations = random.sample(combinations, n_combinations)

    # Create the dictionary mapping process IDs to the chosen combinations
    client_label_map = {}
    for i, combination in enumerate(chosen_combinations):
        
        client_label_map[i] = list(combination)

    print(client_label_map)

    # Ensure that the client_label_map form a complete set from 0 to 9
    unique_values = set()
    for value in client_label_map.values():
        unique_values = unique_values.union(set(value))

    print(unique_values)
    if unique_values != set(range(num_classes)):
      print(f"Invalid 'client_label_map'. Skip.")
      # Count the frequency of each unique value in the client_label_map dictionary
      count_unique_values(client_label_map, num_classes)
      return
      
    count_unique_values(client_label_map, num_classes)
    return client_label_map

        
#Source: https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/sampling.py
def noniid_cifar(dataset, args):
    """
    Sample non-I.I.D client data from CIFAR10 dataset and the data on all clients must contains all 10 labels 
    :param dataset:
    :param num_users:
    :return:
    Colab: https://colab.research.google.com/drive/1hsbUv32k7QIuAmpw8-KiSm-AWTTq97vr#scrollTo=_pFKzjj0pgMt&uniqifier=1+
    """
    
    num_dataset = len(dataset) #40000
    num_users = args.num_users
    if args.sampling == 'oneclassnoniid':
        num_shards = args.num_users #4 the number of shards (partitions) must be equal to the number of classes
    elif args.sampling == 'twoclassnoniid':
        num_shards = args.num_users * 2
    else:
        exit('Error. Non define non-iid sampling method.')
    #partition size
    partition_size  = int(num_dataset/num_shards) #10000 # [50,000 - (classwise * 10)  training imgs]/share -->  40,000 imgs/4 then each shard X 10,000 images
    
    idx_shard = [i for i in range(num_shards)] #[0, 1, 2, 3]
    dict_users = {i: list() for i in range(num_users)} #['0': np.array[], '1': np.array[], ... ]
    
    

    # labels = dataset.train_labels.numpy()
    if args.dataset == "mnist":
        labels = dataset.targets.numpy()
    elif args.dataset == "cifar" or args.dataset == "adv_cifar":
        labels = np.array(dataset.targets) #an array of size 40,000 with class labels from 0 to 9.
    else:
        exit('Error: unrecognized dataset')

    # sort labels again. That's why it might be not necessary
    idxs = np.arange(num_shards*partition_size ) #[0, 1, 2, 3,..., 39999]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()] #it sorts the data by class using numpy.argsort()
    
    idxs = idxs_labels[0]
    labels = idxs_labels[1]

    #for each user
    
    if args.sampling == 'oneclassnoniid':
        for i in range(num_users):  
            rand = np.random.choice(idx_shard, 1, replace=False) # rand 
            # is a np.array has only 1 element.
            idx_shard = list(set(idx_shard) - set(rand)) #update idx_shard
            rand = rand.item() #convert np.array to integer
            rand_set = idxs[rand*partition_size :(rand+1)*partition_size ]
            #divide the idxs list into partitions, and assign [rand*num_imgs, (rand+1)*num_imgs] to the user
            dict_users[i] = rand_set         

    elif args.sampling == 'twoclassnoniid':
        #tensors size (10, 40,000)
        y_train = np.array(dataset.targets)
        
        labels, train_counts = np.unique(y_train, return_counts=True)
        
        target_labels  = np.stack([y_train == y for y in labels])
        print(f'target_labels {target_labels.shape}' )
        if args.num_users == 5:
            client_label_map = {0: [0, 1],
                             1: [2, 3],
                             2: [4, 5],
                             3: [6, 7],
                             4: [8, 9]}
        elif args.num_users == 10:
         # Define the mapping from clients to classes
            client_label_map = {0: [0, 1],
                             1: [2, 3],
                             2: [4, 5],
                             3: [6, 7],
                             4: [8, 9],
                             5: [0, 3],
                             6: [1, 4],
                             7: [2, 5],
                             8: [6, 9],
                             9: [7, 8]}
        elif args.num_users == 8:
            client_label_map = generate_label_map(seed=6, n_combinations=args.num_users)

        elif args.num_users == 16:
            client_label_map = generate_label_map(n_combinations=args.num_users)

        elif args.num_users == 32:
            client_label_map = generate_label_map(n_combinations=args.num_users)

        else:
            exit(f'Error. Non define client_label_map dictionary for {args.num_users}.')

        # for client, labels in client_label_map.items():
        for i in range(num_users):
        
            start_idx = 2 * i #2 class
            end_idx = 2 * (i + 1)
            
            start_idx = client_label_map[i][0] #2 class
            end_idx = client_label_map[i][1] + 1
            
            labels = client_label_map[i]
            
            # print(f'start_idx {start_idx} end_idx {end_idx}')
            print(f'labels {labels}')
            ##the ".where" function returns the "indices" 
            # of the cols where the sum is non-zero, 
            # which correspond to the images that 
            # belong to one of the two categories of 
            # the current client.
            index = np.where(np.sum(target_labels[labels], axis=0))[0]
            print(f'index {index.size}')
    
            perm_split = np.random.permutation(index.size) #shuffle
            index_split_subsample = index[perm_split[:partition_size]]

            # assert (
            #     index_split_subsample.size == 0
            # ), f"Dataset error for client {client} with label {target_labels[labels]}"


            dict_users[i]= index_split_subsample

    #Class dist per party
    if args.sampling == 'oneclassnoniid':
        for idx in range(num_users):
            # Use indices for train/test subsets
            train_indices = dict_users[idx]
            y_train_pi = list()
            for i in list(train_indices):
                y_train_pi.append(dataset[i][1])
            
            y_train_pi = np.array(y_train_pi)
            print_statistics(idx, y_train_pi, 10)
    elif args.sampling == 'twoclassnoniid':
        for i in range(num_users):
            labels, train_counts = np.unique(y_train[dict_users[i]], return_counts=True)
            print(f'Party {i} labels {labels}')
            train_probs = {}
            idx = 0
            for label in labels:
                train_probs[label] = train_counts[idx] / float(len(dict_users[i]))
                idx += 1
            print(f'train_probs {train_probs}')

            # Use indices for train/test subsets
            # train_indices = dict_users[i]
            # y_train_pi = list()
            # for i in range(len(train_indices)):
            #     for j in train_indices[i]:
            #         y_train_pi.append(dataset[j][1])
            # y_train_pi = np.array(y_train_pi)
            # print_statistics(idx, y_train_pi, 10)
    else:
         exit('Error. Non define non-iid sampling method.')


    return dict_users

# In[13]:


def load_dataset(args):
    '''
    output---
    dataset: clean dataset with soft labels instead of integer based labels, 11 classes.
    type: torchvision.datasets.cifar.CIFAR10 with soft labels
    dataset_test: clean dataset with integer based labels, 10 classes
    type: torchvision.datasets.cifar.CIFAR10 
    dg_idx: a list of integer 
    type: Python list
    dg: the global data sharing dataset
    type: torchvision.datasets.cifar.CIFAR10
    dataset_train: the total local data sharing dataset 
    type: torchvision.datasets.cifar.CIFAR10
    dict_users: a dictionary of client id - a list of client's example indices
    type: Python dictionary
    '''
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    
    elif args.dataset == 'cifar':
        transform_train = transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
        
        transform_test = transforms.Compose([ transforms.ToTensor(),
        ])
        dataset = datasets.CIFAR10('../data/cifar', train=True, download=True, 
                                   transform=transform_train)  
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, 
                                        transform=transform_test)
    
    elif args.dataset == 'adv_cifar':
        print('Loaded adv global training data from ' + str(args.adv_global_train_file_name))
        with open(args.adv_global_train_file_name, 'rb') as handle:
            adv_train_dataset = pickle.load(handle)
        print(f'adv_train_dataset {len(adv_train_dataset)}')
        
        print('Loaded adv global test data from ' + str(args.adv_global_test_file_name))
        with open(args.adv_global_test_file_name, 'rb') as handle:
            adv_test_dataset = pickle.load(handle)
        print(f'adv_test_dataset {len(adv_test_dataset)}')
        
        cifar10_mean = [0.4915, 0.4823, 0.4468]
        cifar10_std  = [0.2470, 0.2435, 0.2616]

        test_transform = transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(cifar10_mean, cifar10_std),
                                                ])

        clean_dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=test_transform)
        
        dataset = adv_train_dataset
        dataset_test = adv_test_dataset
    else:
        exit('Error: unrecognized dataset')
    
    # Make a copy of the original dataset
    dg = copy.deepcopy(dataset)
    dataset_train = copy.deepcopy(dataset)
    
    # selected indices for sharing. The rest for training.
    dg_idx, dataset_train_idx = train_dg_split(dataset, args)
    
    # Pixels based on dg_idx, dataset_train_idx
    dg.data, dataset_train.data = dataset.data[dg_idx], dataset.data[dataset_train_idx]
    
    #labels
    if args.dataset == 'cifar' or args.dataset == 'adv_cifar':
        dg.targets.clear()
        dataset_train.targets.clear()
        # Labels based on dg_idx
        dg.targets = [dataset[i][1] for i in dg_idx]

        # Labels based on dataset_train_idx   
        dataset_train.targets = [dataset[i][1] for i in dataset_train_idx]
    else:
        dg.targets, dataset_train.targets = dataset.targets[dg_idx], dataset.targets[dataset_train_idx]

    # sample users, dataset_train
    if args.sampling == 'iid':
        dict_users = iid(dataset_train, args.num_users)
    elif args.sampling == 'noniid':
        dict_users = noniid(dataset_train, args)
    elif args.sampling == 'oneclassnoniid' or args.sampling == 'twoclassnoniid':
        dict_users = noniid_cifar(dataset_train, args)
    else:
        exit('Error: unrecognized sampling')
    
#     # One-hot encoding
#     # Create a list to hold the one-hot encoded targets
#     targets = dataset.targets
#     one_hot_targets = []

#     # Loop through the targets
#     for target in targets:
#         # Create a tensor of zeros with length num_classes + 1
#         one_hot = [0] * (args.num_classes + 1)
#         one_hot[target] = args.soft_label_clean
#         # Set the element at the index corresponding to the target to 1
#         one_hot = [val + 1/(args.num_classes - 1)*(1 - args.soft_label_clean) if i != target else val for i, val in enumerate(one_hot)]
#         one_hot[-1] = 0 #adversarial class
#         # Append the one-hot tensor to the list
#         one_hot_targets.append(one_hot)

#     dataset.targets.clear()
#     dataset.targets = one_hot_targets
    
        
    
    return dataset, dataset_test, dg_idx, dg, dataset_train, dict_users


# In[14]:


dataset, dataset_test, dg_idx, dg, dataset_train, dict_users = load_dataset(args)


# In[15]:


dataset[5][1]


# In[16]:


dataset[5][0].size()


# In[17]:


type(dataset[5][1])


# In[18]:


dict_users.keys()


# In[19]:


# distribute globally shared data (uniform distribution)
for idx in range(args.num_users):
    print(f'party{idx} local: {len(dict_users[idx])}')


# # Build model - Change 2

# In[20]:


def build_model(args, img_size):
    """
    Builds a neural network model based on command line arguments and image size.
    Args:
        args (argparse.Namespace): Command line arguments.
        img_size (tuple): Size of the input images.
    Returns:
        A PyTorch model object.
    """
    if args.model == 'cnn' and (args.dataset == 'cifar' or args.dataset == 'adv_cifar'):
        net_glob = CNN_v2(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNN_v1(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'CNN_v2_resume' and args.filepath is not None:
        net_glob = CNN_v2(args=args)
        print('filepath ', args.filepath)
        weights = torch.load(args.filepath)
        net_glob.load_state_dict(weights) 
        net_glob.to(args.device)
    elif args.model == 'customized_resnet18':
        print('This is customized_resnet18')
        net_glob = ResNet18()
        net_glob = net_glob.to(args.device)
        # net_glob = torch.nn.DataParallel(net_glob)
        # cudnn.benchmark = True
    else:
        net_glob = get_model(args.model, args.pretrained)
        net_glob.to(args.device)
        
    return net_glob


# In[21]:


img_size = dataset_train[0][0].shape
# build model
net_glob = build_model(args, img_size)

print(net_glob)


# ## Gaussian noise (output: 2 Python lists)

# In[23]:


def Gaussian_adversarial_images(args, images, mean, std, min_val = 0, max_val = 1):
    '''
    Inputs:
    images: torch.Size([batch size, 3, 32, 32])
    type: Torch.Tensor
    mean: the mean of the Gaussian noise to be added
    std: the standard deviation of the Gaussian noise to be added
    min_val: the minimum value of the images (default: 0)
    max_val: the maximum value of the images (default: 1)
    
    Output:
    images_list: a list of (C x H x W) channel first clean images 
    type: a list of Torch objects
    images_adv_list: a list of (C x H x W) channel first adversarial images 
    type: a list of Torch objects
    '''
    images_adv_list = []
    images_list = []
        
    #images = Variable(images,requires_grad = True)
    noise = torch.normal(mean, std, size=images.shape).cuda()
    
    # add the noise to the images
    images_adv = torch.clamp(images + noise, min=min_val, max=max_val)
    
    images_adv_list.extend(images_adv) #a list of Tensor
    images_list.extend(images) #a list of Tensor
    
    return images_list, images_adv_list


# ## FGSM (output: 2 Python lists)

# In[24]:


def FGSM(args, model, criterion, images, labels, epsilon = 8/255, min_val = 0,max_val = 1):
    '''
    Inputs:
    images: torch.Size([batch size, 3, 32, 32])
    type: Torch.Tensor
    labels: torch.Size([6, 11])
    type: torch.Tensor
    '''
    
    '''
    Output:
    images_list: a list of (C x H x W) channel first clean images 
    type: a list of Torch objects
    images_adv_list: a list of (C x H x W) channel first adversarial images 
    type: a list of Torch objects
    '''
    images_adv_list = []
    images_list = []

    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
        
        #images = Variable(images,requires_grad = True)
    images.requires_grad = True
    
    outputs = model(images)
    loss =criterion(outputs,labels)

    model.zero_grad()
    if images.grad is not None:
        images.grad.data.fill_(0)
    loss.backward()

    grad = torch.sign(images.grad.data) # Take the sign of the gradient.
    images_adv = torch.clamp(images.data + epsilon*grad,min_val,max_val)     # x_adv = x + epsilon*grad

    # adverserial_images.extend((images_adv).cpu().data.numpy())
    # images_list.extend(images.cpu().data.numpy())
    
    images_adv_list.extend(images_adv) #a list of Tensor
    images_list.extend(images) #a list of Tensor
    
    return images_list, images_adv_list


# ## FGSM noise

# In[25]:


def FGSM_noise(args, model, criterion, images, labels, epsilon = 8/255,
               mean=0, sigma=0.1, min_val = 0, max_val = 1):
        
    images_adv_list = []
    images_list = []
    # with torch.enable_grad():
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
    images.requires_grad = True

    outputs = model(images)
    loss=criterion(outputs,labels)

    model.zero_grad()
    if images.grad is not None:
        images.grad.data.fill_(0)
    loss.backward()

    grad = torch.sign(images.grad.data) # Take the sign of the gradient.
    images_adv = torch.clamp(images.data + epsilon*grad,min_val,max_val)     # x_adv = x + epsilon*grad
    

    noise = torch.normal(mean, sigma,size=images_adv.shape).to(args.device)
    images_adv_noise = images_adv + noise
    images_adv_noise = torch.clamp(images_adv_noise, min_val, max_val)
    
    images_adv_list.extend(images_adv) #a list of Tensor
    images_list.extend(images) #a list of Tensor
    
    # noise_L2_norm = torch.norm(noise)
    # print(f'noise_L2_norm: {type(noise_L2_norm)}') 
    # print(f'noise_L2_norm: {noise_L2_norm}') 
          
    return images_list, images_adv_list    


# ## LinfPGDAttack

# In[26]:


class LinfPGDAttack(object):
    '''
    Input: tensors
    Output: tensors
    '''
    def __init__(self, model, eps, nb_iter, eps_iter, clip_min, clip_max):
        self.model = model
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.clip_min = clip_min
        self.clip_max = clip_max


    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.eps, self.eps)
        for i in range(self.nb_iter):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.eps_iter * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.eps), x_natural + self.eps)
            x = torch.clamp(x, self.clip_min, self.clip_max)
        return x


# ## visualize_adversarial_images - Long test, remove y_preds

# In[27]:


def visualize_adversarial_images(args, round_cur, adversarial_images, y_preds, y_preds_adv, 
                                 images_list, label_list, epsilon):
    if args.dataset == 'cifar':
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        

        adversarial_images = np.array(adversarial_images)
        y_preds = np.array(y_preds)
        y_preds_adv = np.array(y_preds_adv)
        images_list = np.array(images_list)

        c = adversarial_images - images_list  # Verify whether the max diff between the image and adversarial image is epsilon or not
        if np.any(np.abs(c.max()) > epsilon + 0.01):
            print('the difference is more than the epsilon')

        
        mean = np.array([0.5, 0.5, 0.5])
        mean = mean[:, None, None]
        std = np.array([0.5, 0.5, 0.5])
        std = std[:, None, None]
        

        # Get index of all the images where the attack is successful
        attack = (y_preds != y_preds_adv)
        indexes = np.where(attack == True)[0]

        # Plot the images
        plt_idx = 0
        while plt_idx < 2:
            idx = np.random.choice(indexes)
            img = images_list[idx]
            adv_img = adversarial_images[idx]

            img = img * std + mean
            img = np.transpose(img, (1, 2, 0))
            img = img.clip(0, 1)

            adv_img = adv_img * std + mean
            adv_img = np.transpose(adv_img, (1, 2, 0))
            adv_img = adv_img.clip(0, 1)

            noise = adv_img - img
            noise = np.absolute(10 * noise)  # Noise is multiplied by 10 for visualization purpose
            noise = noise.clip(0, 1)

            if y_preds[idx] != y_preds_adv[idx]:
                disp_im = np.concatenate((img, adv_img, noise), axis=1)
                ax = plt.subplot(1, 2, plt_idx + 1)
                ax.set_title("pred: {}, adv:{}".format(classes[y_preds[idx]], classes[y_preds_adv[idx]]))
                plt.imshow(disp_im)
                plt.xticks([])
                plt.yticks([])
                plt_idx += 1
                print("True Label: ", classes[label_list[idx]], " ", "Predicted Label:", classes[y_preds[idx]], " ",
                      "Adversarial Label:", classes[y_preds_adv[idx]])
        
        name_file = f'./save/{args.fed}_{args.eps_FGSM}_{args.dataset}_{args.model}_{args.local_ep}_nParties_{len(idxs_users)}_{args.sampling}_{args.classwise}_{args.alpha}_round_cur_{round_cur}'
        name_file_1 = name_file + '_visualization.pdf'
        plt.savefig(name_file_1)
        plt.close()


# # Test image methods

# ### Stoppping criterion

# In[28]:


def calculate_expected_perturbation_proportion(args, adversarial_images, images_list, delta = 1e-10):
    
    adversarial_images = torch.stack(adversarial_images).cpu().detach().numpy()
    # print(adversarial_images.size())
    images_list = torch.stack(images_list).cpu().detach().numpy() #detach
        
    adversarial_images = np.array(adversarial_images)
    images_list = np.array(images_list)
    
    rho = np.zeros(len(images_list)) # record the size of perturbation (2-norm)
    for i in range(len(images_list)):
        diff = images_list[i]-adversarial_images[i]
        
        #L2-norm
        if args.dataset == 'cifar':
            diff = diff.reshape((32*32,3))
            
        else:
            diff = diff.reshape((28*28,1))

        rho[i] = LA.norm(diff)/(LA.norm(images_list[i]) + delta)
    #sum
    rho =  np.sum(rho)
    return rho


# ### Test method 1, no noise

# In[29]:


def test_img(net_g, datatest, idxs, args, criterion):
    net_g.eval()
    
    # testing
    test_loss = 0
    correct = 0
    adv_test_loss = 0
    adv_correct = 0
    adv_correct_2 = 0
    misclassified = 0 #change decision
    
    #visualize
    exp_adv_noise = 0
    y_preds = []
    y_preds_adv = []
    test_images = []
    test_label = []
    adverserial_images = []

    data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        #clean
        with torch.no_grad():
        
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device) 

            log_probs = net_g(data)
            # sum up batch loss
            # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        #Gaussian noise
        
        #FGSM method
        images_list, images_adv_list = FGSM(args, net_g, criterion, data, target,
                                                   args.eps_FGSM, args.clip_min, args.clip_max) #The adversarial_images list contains a series of perturbed images
        
        images_adv = torch.stack(images_adv_list) #return a tensor of size (args.bs, 3, 32, 32)
        images_adv = images_adv.to(args.device)
        
        adv_noise = calculate_expected_perturbation_proportion(args, 
                                                               images_adv_list, 
                                                               images_list, 
                                                               1e-10)
        
        with torch.no_grad():
            
            log_probs = net_g(images_adv)
            # sum up batch loss
            # adv_test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            
            # get the index of the max log-probability
            adv_y_pred = log_probs.data.max(1, keepdim=True)[1]
            adv_correct += adv_y_pred.eq(target.data.view_as(adv_y_pred)).long().cpu().sum()
            
        misclassified += (y_pred != adv_y_pred).sum().item()
        y_preds.extend(y_pred.cpu().data.numpy())
        y_preds_adv.extend(adv_y_pred.cpu().data.numpy())
        test_images.extend(data.cpu().data.numpy())
        test_label.extend(target.cpu().data.numpy())
        adverserial_images.extend((images_adv).cpu().data.numpy())
        exp_adv_noise += adv_noise
        
    #test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    #adv_test_loss /= len(data_loader.dataset)
    adv_acc = 100.00 * adv_correct / len(data_loader.dataset)
    
    #average the noise
    exp_adv_noise = np.sum(exp_adv_noise)/len(data_loader.dataset)
    #visualize
    # visualize_adversarial_images(args, 0, adverserial_images, y_preds, y_preds_adv, 
    #                              test_images, test_label, args.eps_FGSM)
    
    #sum and average adv_noise_proportion
    if args.verbose:
        print('\nTest set: \nAccuracy: {}/{} ({:.2f}%)\n'.format(
                correct, len(data_loader.dataset), accuracy))
        
        
        print('\nAdversarial Test set as a classifier: \nAdv Accuracy: {}/{} ({:.2f}%)\n'.format(
                adv_correct, len(data_loader.dataset), adv_acc))
        
        print('\nNumber of correct classified clean examples(as compared to clean predictions): {}/{}'.format(correct, len(data_loader.dataset)))
        print('\nNumber of correct classified adversarial examples(as compared to clean predictions): {}/{}'.format(adv_correct, len(data_loader.dataset)))
        print('\nNumber of attack success: {}/{}'.format(misclassified, len(data_loader.dataset)))
    
    
        
    return accuracy, adv_acc, exp_adv_noise


# ### Test method 2 with noise

# #### evaluate_adversarial_images

# In[30]:


def evaluate_adversarial_images(args, images, labels, net_g, criterion, attack_method='Gaussian'):
    
    correct = 0
    if attack_method == 'Gaussian':
    # Generate adversarial images using Gaussian perturbation
       images_list, images_adv_list = Gaussian_adversarial_images(args, images,
                                                                      args.mean, args.sigma, args.clip_min, args.clip_max)
    
    elif attack_method == 'FGSM':
        adv_correct = 0
        images_list, images_adv_list = FGSM(args, net_g, criterion, images, labels,
                                                       args.eps_FGSM, args.clip_min, args.clip_max)
    elif attack_method == 'FGSM_noise':
        adv_correct = 0
        images_list, images_adv_list = FGSM(args, net_g, criterion, images, labels,
                                                       args.eps_FGSM, args.clip_min, args.clip_max)
        
        adversarial_images = torch.stack(images_adv_list)
        #add noise
        (_), images_adv_list = Gaussian_adversarial_images(args, adversarial_images,
                                                                      args.mean, args.sigma, args.clip_min, args.clip_max)
    else:
        exit('Unrecognized attack method')
        
 
    images_adv = torch.stack(images_adv_list)    
    images_adv = images_adv.to(args.device)
    
    # print(f'images_adv.shape {images_adv.shape}')
    log_probs = net_g(images_adv)
    
    # print(f'log_probs.shape {log_probs.shape}')
    
    adv_y_pred = log_probs.data.max(1, keepdim=True)[1]
    
    # print(f'adv_y_pred.shape {adv_y_pred.shape}')
    
    # print(f'labels {labels.shape}')
    
    correct += adv_y_pred.eq(labels.data.view_as(adv_y_pred)).long().cpu().sum()
    if attack_method != 'Gaussian':
        adv_noise = calculate_expected_perturbation_proportion(args, 
                                                               images_adv_list, 
                                                               images_list, 
                                                               1e-10)
        
        adv_correct += (adv_y_pred == 10).int().cpu().sum()
        #return adversarial attack
        return adv_y_pred, correct, adv_correct, adv_noise
    #return Gaussian noise
    return adv_y_pred, correct
    




# #### Test method 2

# In[31]:


def test_img_noise(net_g, datatest, idxs, args, criterion):
    net_g.eval()
    
    # testing
    # test_loss = 0
    correct = 0
    FGSM_correct_rand = 0
    
    Gaussian_misclassified = 0 #change decision
    FGSM_rand_misclassified = 0
    
    #visualize
    FGSM_total_adv_noise  = 0
    FGSM_avg_adv_noise = 0.0
    #
    y_preds = []
    Gaussian_pred_list = []
    FGSM_pred_rand_list = []
    #
    test_images = []
    test_label = []
    #
    FGSM_adv_noise = 0
    
    data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.bs)
    l = len(data_loader)
    # Evaluate the classifier on the normal test set 
    for idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
        
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device) 

            log_probs = net_g(data)
            # sum up batch loss
            # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            
    ######################################### Gaussian noise ##################################
            Gaussian_pred, Gaussian_correct = evaluate_adversarial_images(args, data, 
                                                            target, 
                                                            net_g, 
                                                            criterion, 
                                                            'Gaussian')

    ##############################################FGSM method##################################
        
        ##############################################FGSM-noise method##################################
            
        images_list, images_adv_list_noisy = FGSM_noise(args, net_g, criterion, data, 
                                                  target, args.eps_FGSM,
                                                   args.mean, 
                                                  args.sigma, 
                                                  args.clip_min, 
                                                  args.clip_max)
        
        images_adv_noisy = torch.stack(images_adv_list_noisy) #return a tensor of size (args.bs, 3, 32, 32)
        images_adv_noisy = images_adv_noisy.to(args.device)
        
        adv_noise = calculate_expected_perturbation_proportion(args, 
                                                               images_list, 
                                                               images_adv_list_noisy, 
                                                               1e-10)
        with torch.no_grad():
            
            log_probs = net_g(images_adv_noisy)
            # sum up batch loss
            # adv_test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            
            # get the index of the max log-probability
            FGSM_pred_rand = log_probs.data.max(1, keepdim=True)[1]
            FGSM_correct_rand += FGSM_pred_rand.eq(target.data.view_as(FGSM_pred_rand)).long().cpu().sum()
        
        
        ###############################################################################
        Gaussian_misclassified += (y_pred != Gaussian_pred).sum().item() #proof of vulnerable
        FGSM_rand_misclassified += (y_pred != FGSM_pred_rand).sum().item() #change decision
        
        ###############################################################################
        y_preds.extend(y_pred.cpu().data.numpy())
        Gaussian_pred_list.extend(Gaussian_pred.cpu().data.numpy())
        FGSM_pred_rand_list.extend(FGSM_pred_rand.cpu().data.numpy())
        
        ###############################################################################
        test_images.extend(data.cpu().data.numpy())
        test_label.extend(target.cpu().data.numpy())
        
        FGSM_total_adv_noise += adv_noise
        
    #calculates the accuracy of the model on the clean test examples
    accuracy = 100.00 * correct / len(data_loader.dataset)
    
    #calculates the accuracy of the model on the Gaussian adversarial examples
    Gaussian_acc = 100.00 * Gaussian_correct / len(data_loader.dataset)
        
    #FGSM with noise
    FGSM_acc_noise = 100.00 * FGSM_correct_rand / len(data_loader.dataset)
    
    #average the noise
    FGSM_avg_adv_noise = np.sum(FGSM_total_adv_noise)/len(data_loader.dataset)
    #visualize
    # visualize_adversarial_images(args, 0, adverserial_images, y_preds, y_preds_adv, 
    #                              test_images, test_label, args.eps_FGSM)
    
    #sum and average adv_noise_proportion
    if args.verbose:
        print('\nTest set: \nAccuracy on benign test examples: {}/{} ({:.2f}%)\n'.format(
                correct, len(data_loader.dataset), accuracy))
        
        print('\nTest set: \nAccuracy on Gaussin noise test examples: {}/{} ({:.2f}%)\n'.format(
                Gaussian_correct, len(data_loader.dataset), Gaussian_acc))
        
        print('\nTest accuracy for noisy FGSM: \nAdv Accuracy: {}/{} ({:.2f}%)\n'.format(
                FGSM_acc_noise, len(data_loader.dataset), FGSM_acc_noise))
        
        print('\nNumber of attack success: {}/{}'.format(FGSM_rand_misclassified, len(data_loader.dataset)))
        #Jing's evaluation
        
    return accuracy, Gaussian_acc, FGSM_acc_noise, FGSM_avg_adv_noise


# In[32]:


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# # NOTE:  Test only the first 100 images

# In[33]:





# ## soft_label_adversarial_images

# In[38]:


# def soft_label_adversarial_images(args, adversarial_images, sl_tau, sl_alpha, 
#                                 images_list, label_list):
    
#     adversarial_images = torch.stack(adversarial_images).cpu()
#     # print(adversarial_images.size())
#     images_list = torch.stack(images_list).cpu().detach().numpy() #detach
#     label_list = label_list.cpu()
    
#     adversarial_images = np.array(adversarial_images)
#     images_list = np.array(images_list)
#     label_list = np.array(label_list)
    
#     label_list_soft = np.copy(label_list)

#     perturbation = np.zeros(len(label_list)) # record the size of perturbation (2-norm)
#     for i in range(len(label_list)):
#         diff = images_list[i]-adversarial_images[i]
        
#         #L2-norm
#         if args.dataset == 'cifar':
#             diff = diff.reshape((32*32,3))
#             perturbation[i] = LA.norm(diff)/(32*math.sqrt(3))
#         else:
#             diff = diff.reshape((28*28,1))
#             perturbation[i] = LA.norm(diff)/(28)
#         #For each image
#         true_label = np.argmax(label_list_soft[i])
#         if perturbation[i] < sl_tau:
#             one_hot = [0] * (args.num_classes + 1)
#             one_hot[true_label] = sl_alpha
#             one_hot = [1/(args.num_classes)*(1 - sl_alpha) if j != true_label else val for j, val in enumerate(one_hot)]
#             # print(f'label_list_soft[i] {label_list_soft[i].shape}')
#             # print(f'one_hot {one_hot}')
#             label_list_soft[i] = np.array(one_hot)
#         else:
#             beta = sl_alpha/2 + (1-sl_alpha)/11
#             gamma = beta
#             one_hot = [0] * (args.num_classes + 1)
#             one_hot[true_label] = beta
#             one_hot = [1/(args.num_classes - 1)*(1 - sl_alpha) if j != true_label else val for j, val in enumerate(one_hot)]
#             one_hot[-1] = gamma
#             # label_list_soft[i, :] = (1/9)*(1 - beta - gamma) #evenly distributes the remaining probability mass among the other classes.
#             label_list_soft[i] = np.array(one_hot)
        
#    return label_list_soft.tolist()


# ## Soft label Cross entropy loss

# In[77]:

def generate_adversarial_images(args, images, labels, net_glob, criterion):
    
    # Generate adversarial images using Gaussian perturbation
    images_list, adversarial_images = Gaussian_adversarial_images(args, images,
                                                                  args.mean, args.sigma, 
                                                                  args.clip_min,
                                                                  args.clip_max)
    

    adversarial_images = torch.stack(adversarial_images)
    # print(f'Gaussian noise adversarial_images size: {adversarial_images.shape[0]}')
    
    # Generate adversarial images using Fast Gradient Sign Method (FGSM)
    adversary = LinfPGDAttack(net_glob, eps=args.eps, nb_iter=args.nb_iter, 
                                            eps_iter=args.eps_iter,
                                            clip_min=args.clip_min,
                                            clip_max=args.clip_max)
    
    adversarial_images_PGD = adversary.perturb(images, labels)

    adversarial_images = torch.cat([adversarial_images, adversarial_images_PGD], dim=0)
    # print(f'PGD noise adversarial_images size: {adversarial_images_PGD.shape[0]}')
    
    # print(f'the augmented training set size: {adversarial_images.shape[0]}')

    return adversarial_images


# ### Test generate_adversarial_images - Issue*

# In[41]:



# # Utils

# In[35]:


def save_model(start_time, args, net_glob, idxs_users):
    try:
        now = datetime.now()
        print("Total time for the training: {} seconds ---".format(now - start_time))
        now = start_time.strftime("%Y-%m-%dT%H-%M-%S")

        file = f'{args.fed}_{args.eps_FGSM}_{args.dataset}_{args.model}_${args.rounds}_{args.local_ep}_nParties_{len(idxs_users)}_{args.sampling}__{args.classwise}_{args.alpha}_model'
        model_name = '{}_{}.pt'.format(file, now)
        filepath = './save/'
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        f_model = os.path.join(filepath, model_name)
        torch.save(net_glob.state_dict(), f_model)
        
        print('The final model saved to: ', f_model)
        
        return f_model
    except Exception as e:
        print(f'Error saving model: {e}')


# In[36]:


def save_model_performance(args, start_time, rounds_test_accuracy,
                           rounds_train_loss, 
                           rounds_adv_test_accuracy, 
                           rounds_adv_test_accuracy_2,
                           idxs_users):
    try:
        output = {}
        output['rounds_test_accuracy'] = rounds_test_accuracy
        output['rounds_train_loss'] = rounds_train_loss
        output['rounds_adv_test_accuracy'] = rounds_adv_test_accuracy
        output['rounds_adv_test_accuracy_2'] = rounds_adv_test_accuracy_2

        temp='./save/'
        

        filename = f'{args.fed}_{args.eps_FGSM}_{args.dataset}_{args.model}_${args.rounds}_{args.local_ep}_nParties_{len(idxs_users)}_{args.sampling}_{args.classwise}_{args.alpha}'
        filename = '{}_{}.out'.format(filename, start_time.strftime("%Y-%m-%dT%H-%M-%S"))
        filepath = os.path.join(temp, filename)
        print('filepath ', filepath)
        outfile = open(filepath,'wb')
        pickle.dump(output, outfile)

        print('The output file saved to: ', filepath)
    except Exception as e:
            print(f'Error saving model performance: {e}')


# In[37]:


def read_model_performance(args, temp, idxs_users):
    try:
        temp='./save/'
        filename = f'{args.fed}_{args.eps_FGSM}_{args.dataset}_{args.model}_{args.local_ep}_nParties_{idxs_users}_{args.sampling}_{args.classwise}_{args.alpha}_output.out'
        filepath = os.path.join(temp, filename)
        print('filepath ', filepath)
        
        output = pickle.load(open(filepath, "rb"))
        
        rounds_test_accuracy = output['rounds_test_accuracy']
        rounds_train_loss = output['rounds_train_loss']
        rounds_adv_test_accuracy = output['rounds_adv_test_accuracy'] 
        rounds_adv_test_accuracy_2 = output['rounds_adv_test_accuracy_2']
        
        print(len(rounds_test_accuracy))
        print(len(rounds_adv_test_accuracy_2))
        print(len(rounds_train_loss))
        ################### training loss ######################
#         label = args.fed
#         plt.xlabel("Global Iterations")
#         plt.ylabel(f"Train loss")
#         # plt.set_yscale('log')
#         plt.title(f"Centralized AT - Dataset: {args.dataset} - Model: {args.model} - Soft labelling" )
#         plt.plot(rounds_train_loss, label='soft label')
        
#         plt.legend(loc="center right")
#         plt.show()
#         plt.close()
        ###################################################
        label = args.fed
        plt.xlabel("Global Iterations")
        plt.ylabel(f"Test acc")
        # plt.set_yscale('log')
        plt.title(f"Centralized AT - Dataset: {args.dataset} - Model: {args.model} - Soft labelling" )
        plt.plot(rounds_test_accuracy, label='clean acc')
        plt.plot(rounds_adv_test_accuracy, label='additional class acc')
        plt.plot(rounds_adv_test_accuracy_2, label='robust acc')
        
        plt.legend(loc="lower right")
        plt.show()
        plt.close()
        
    except Exception as e:
            print(f'Error loading model performance: {e}')


# In[38]:


import re

def parse_output_log(args, file_name, save_file):
    train_loss = []
    test_accuracy = []
    adv_accuracy = []
    
    for line in open(file_name, 'r'):
        # print(line) 
        search_train_loss = re.search(r'Total adversarial train loss: ([0-9].\d+)', line, re.M|re.I)
        if search_train_loss:
            # print(search_train_loss)
            # print(search_train_loss.group(1))
            train_loss.append(float(search_train_loss.group(1)))
        
        # search for test accuracy
        search_test_accu = re.search(r'Number of correct classified clean examples\(as compared to clean predictions\): (\d+)/\d+', line, re.M|re.I)
        if search_test_accu:
            val = float(search_test_accu.group(1))
            # print(val)
            test_accuracy.append(val/100)
            # if(len(test_accuracy) == 5):
            #     break
        # search for adversarial accuracy
        search_adv_accu = re.search(r'Adv Accuracy: \d+/\d+ \((\d+\.\d+)%\)', line, re.M|re.I)
        if search_adv_accu:
            val = float(search_adv_accu.group(1))
            if val > 1.0:
                adv_accuracy.append(float(search_adv_accu.group(1)))
        
    print(f'train_loss {len(train_loss)}')
    print(f'test_accuracy {len(test_accuracy)}')
    print(f'adv_accuracy {len(adv_accuracy)}')
    
    # plot train loss
    plt.figure(figsize=(8, 4))
    plt.xlabel("Global Iterations")
    plt.ylabel(f"Train loss")
    plt.title(f"Centralized AT - Dataset: {args.dataset} - Model: {args.model} - Soft labelling" )
    plt.plot(train_loss, label='train loss')
    plt.legend(loc="lower right")
    filepath_1 = save_file + '/AT_sl_simplified_train_loss.png'
    plt.savefig(filepath_1)
    # plt.show()
    plt.close()
    
    # plot test accuracy
    plt.xlabel("Global Iterations")
    plt.ylabel(f"Test acc")
    # plt.set_yscale('log')
    plt.title(f"Centralized AT - Dataset: {args.dataset} - Model: {args.model} - Soft labelling" )
    plt.plot(test_accuracy, label='clean acc') #scale: 0.8
    plt.plot(adv_accuracy, label='robust acc') #percentage

    plt.legend(loc="lower right")
    # plt.show()
    filepath_2 = save_file + '/AT_sl_simplified_test_acc.png'
    plt.savefig(filepath_2)
    plt.close()
    
    
    return train_loss, test_accuracy


# # Update local weights using adversarial examples and soft labels - Change 7

# In[39]:


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# In[40]:


def adjust_learning_rate(args, optimizer, epoch):
        lr = args.lr
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# ### Adversarial_ModelUpdateAdversarial_ModelUpdate

# In[41]:


class Adversarial_ModelUpdate(object):
    def __init__(self, args, client_id, local_ep=1, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), 
                                    batch_size=self.args.local_bs, shuffle=True)
        self.local_ep = local_ep
        self.client_id = client_id
        
    
    def smooth_label(self, y):
        '''
        y: integer-base
        '''
        # convert y to one-hot encoding
        y_onehot = F.one_hot(y, self.args.num_classes)
        # smooth the one-hot encoding
        y_smooth = (self.args.soft_label_clean) * y_onehot + (1 - self.args.soft_label_clean) / (self.args.num_classes)
        # convert back to class labels
        y_smooth = torch.argmax(y_smooth, dim=1)
        return y_smooth

    def train(self, pid, round_cur, local_net, net):
        
        net.train()
        
        # train and update
        # weight_decay=self.args.l2_lambda
        if self.args.l2_lambda != 0:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, weight_decay=self.args.l2_lambda, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []

        if self.args.sys_homo: 
            local_ep = self.local_ep
        else:
            local_ep = randint(self.args.min_le, self.args.max_le) 
        
        for iter in range(local_ep):
            print(f'\n[ Current round {round_cur} Train epoch: {iter} ]')
            batch_loss = []
            correct = 0
            train_size = 0

            adjust_learning_rate(self.args, optimizer, iter)
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                adversarial_images = generate_adversarial_images(self.args, 
                                                                    images, 
                                                                    labels, 
                                                                    net, 
                                                                    self.loss_func)
    
                adversarial_images = adversarial_images.to(self.args.device)
    
                labels_smooth = self.smooth_label(labels) #Test on Gaivi
                
                soft_label_list = torch.cat([labels_smooth, labels_smooth], dim=0)

                soft_label_list = soft_label_list.to(self.args.device)

                net.zero_grad()
                log_probs = net(adversarial_images) #model predictions, no torch concaternation
                loss = self.loss_func(log_probs, soft_label_list) #labels here 11 class and clean examples
                    
                loss.backward()
                
                optimizer.step()
                
                # get the index of the max log-probability
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(soft_label_list.data.view_as(y_pred)).long().cpu().sum()
                train_size += adversarial_images.shape[0]
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Round : {} Party {}: Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        round_cur, pid, iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            print('\nTotal adversarial train accuarcy:', 100. * correct / train_size)
            print('\nTotal adversarial train loss:', sum(batch_loss)/len(batch_loss))
           
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


# # Update global weights

# In[42]:


def FedAvg(w, args, c_global=None, res_caches=None):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        tmp = torch.zeros_like(w[0][k], dtype = torch.float32).to(args.device)
        for i in range(len(w)):
            tmp += w[i][k]
        tmp = torch.true_divide(tmp, len(w))
        w_avg[k].copy_(tmp)
    if args.fed == 'scaffold' and c_global is not None and res_caches is not None:
        if args.all_clients:
            client_num_per_round = args.num_users
        else:
            client_num_per_round = max(int(args.frac * args.num_users), 1)
        
        # update global model
        avg_weight = torch.tensor(
                        [
                            1 / client_num_per_round
                            for _ in range(client_num_per_round)
                        ],
                        device=args.device,
                    ) #by number of selected clients per round, not dependent on the local data size
        # y_pred = net_glob(inputs).cpu()
        # print(y_pred.detach().numpy())
        # print(f'avg_weight: {avg_weight.cpu().detach().numpy()}')
        c_delta_cache = list(zip(*res_caches))
        # print(f'c_delta_cache {c_delta_cache}')
        # update global control
        for c_g, c_del in zip(c_global, c_delta_cache):
            # print(f'before c_g {c_g.cpu().detach().numpy()}')
            c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1) #delta_c = sum of avg_weight * delta_c_i
            # print(f'c_del: {c_del.cpu().detach().numpy()}')
            c_g.data += (
                client_num_per_round / args.num_users
            ) * c_del #c_global = |S| / N * c_delta
            # print(f'c_g {c_g.cpu().detach().numpy()}')
        return w_avg, c_global
    return w_avg


# ## Initialization using Data Sharing - Change 5

#define the loss function
criterion = torch.nn.CrossEntropyLoss()


#Load model
net_glob = build_model(args, img_size)
print('filepath ', args.filepath)
weights = torch.load(args.filepath)
net_glob.load_state_dict(weights) 
net_glob.to(args.device)

#FGSM
test_idxs = np.arange(len(dataset_test)) ###
acc_test, adv_acc_test, (_) = test_img(net_glob, 
                                        dataset_test, 
                                        test_idxs, 
                                        args, 
                                        criterion)

#FGSM noise
accuracy, Gaussian_acc, FGSM_acc_noise, exp_adv_noise = test_img_noise(net_glob, 
                                                            dataset_test,
                                                            test_idxs,
                                                            args, 
                                                            criterion)


#C&W
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import CarliniLInfMethod as cw

optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, weight_decay=args.l2_lambda, momentum=args.momentum)

#Create the ART classifier
classifier = PyTorchClassifier(
    model=net_glob,
    clip_values=(args.clip_min, args.clip_max),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=args.num_classes,
)

test_idxs = np.arange(100)
test_data_loader = DataLoader(DatasetSplit(dataset_test, test_idxs), batch_size=args.bs)

#FGSM
from art.attacks.evasion import FastGradientMethod
adv_crafter = FastGradientMethod(estimator=classifier, eps=0.031)

np.random.seed(args.seed)

def evaluate_attack(test_data_loader, 
    classifier, 
    adv_crafter,
    attack_name,
    mode='noise'):
    correct = 0
    
    # iterate over the test data loader
    for idx, (data, target) in enumerate(test_data_loader):
        data = data.cpu().numpy()
        target = target.cpu().numpy()

        # generate adversarial examples
        x_adv_batch = adv_crafter.generate(x=data)

        if mode == 'noise':
            noise = np.random.normal(args.mean, args.sigma, size=x_adv_batch.shape).astype(np.float32)

            x_adv_batch = x_adv_batch + noise
        
            x_adv_batch = np.clip(x_adv_batch, args.clip_min, args.clip_max)
        

        log_probs = classifier.predict(x_adv_batch, training_mode=False) 
        y_pred = np.argmax(log_probs, axis=1)
        correct += np.sum(y_pred == target)
    
    # set the classifier back to training mode
    if mode == 'noise':
        print(f'{attack_name}-noise: adversarial correct {correct} / {len(test_data_loader)}')
    else:
        print(f'{attack_name}-no noise: adversarial correct {correct} / {len(test_data_loader)}')


evaluate_attack(test_data_loader, 
    classifier, 
    adv_crafter,
    'FGSM',
    mode='clean')

evaluate_attack(test_data_loader, 
    classifier, 
    adv_crafter,
    'FGSM',
    mode='noise')

#BIM
from art.attacks.evasion import BasicIterativeMethod as bim

adv_crafter = bim(classifier, eps=0.1, 
                                eps_step=0.1/3,
                                max_iter=40)

evaluate_attack(test_data_loader, 
    classifier, 
    adv_crafter,
     'BIM',
    mode='clean')

evaluate_attack(test_data_loader, 
    classifier, 
    adv_crafter,
    'BIM',
    mode='noise')

#PGD

from art.attacks.evasion import ProjectedGradientDescentPyTorch as pgd_Torch

adv_crafter = pgd_Torch(classifier, eps=args.eps, 
                                eps_step=args.eps_iter,
                                max_iter=args.nb_iter)

evaluate_attack(test_data_loader, 
    classifier, 
    adv_crafter,
    'PGD',
    mode='clean')

evaluate_attack(test_data_loader, 
    classifier, 
    adv_crafter,
    'PGD',
    mode='noise')

#cw
adv_crafter = cw(classifier, targeted=False, batch_size=args.bs, confidence=0.05, verbose=False)

evaluate_attack(test_data_loader, 
    classifier, 
    adv_crafter,
    'CW',
    mode='clean')

evaluate_attack(test_data_loader, 
    classifier, 
    adv_crafter,
    'CW',
    mode='noise')

#DeepFool
from art.attacks.evasion import DeepFool
adv_crafter = DeepFool(classifier, batch_size=args.bs, verbose=False)

evaluate_attack(test_data_loader, 
    classifier, 
    adv_crafter,
    'DeepFool',
    mode='clean')

evaluate_attack(test_data_loader, 
    classifier, 
    adv_crafter,
    'DeepFool',
    mode='noise')



