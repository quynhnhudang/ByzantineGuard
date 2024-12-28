import pickle
import torch 
from torchvision import datasets, transforms
import copy

from utils.distribute import uniform_distribute, train_dg_split
from utils.sampling import iid, noniid


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
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)  
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    
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
    else:
        exit('Error: unrecognized sampling')
    
    # One-hot encoding
    # Create a list to hold the one-hot encoded targets
    targets = dataset.targets
    one_hot_targets = []

    # Loop through the targets
    for target in targets:
        # Create a tensor of zeros with length num_classes + 1
        one_hot = [0] * (args.num_classes + 1)
        one_hot[target] = args.soft_label_clean
        # Set the element at the index corresponding to the target to 1
        one_hot = [val + 1/(args.num_classes - 1)*(1 - args.soft_label_clean) if i != target else val for i, val in enumerate(one_hot)]
        one_hot[-1] = 0 #adversarial class
        # Append the one-hot tensor to the list
        one_hot_targets.append(one_hot)

    dataset.targets.clear()
    dataset.targets = one_hot_targets
    
        
    
    return dataset, dataset_test, dg_idx, dg, dataset_train, dict_users

def get_dataset(dir, name):

    if name=='mnist':
        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
        
    elif name=='cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = datasets.CIFAR10(dir, train=True, download=True,
                                        transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
    
    elif name=='adv_cifar':
        adv_global_train_file_name = '/work_bgfs/xionggroup/2022/adversarial-train-ParallelComp/transfer_learning/DenseNet/Small_input_size/eps_0.1/adv_train_cifar10_fgsm.dataset'
        print('Loaded adv global training data from ' + str(adv_global_train_file_name))
        with open(adv_global_train_file_name, 'rb') as handle:
            train_dataset = pickle.load(handle)

        print(f'adv_train_dataset {len(train_dataset)}')
        ############################################################################################
        adv_global_test_file_name = '/work_bgfs/xionggroup/2022/adversarial-train-ParallelComp/transfer_learning/DenseNet/Small_input_size/eps_0.1/adv_test_cifar10_fgsm.dataset'
        print('Loaded adv global test data from ' + str(adv_global_test_file_name))

        with open(adv_global_test_file_name, 'rb') as handle:
            eval_dataset = pickle.load(handle)
        print(f'adv_test_dataset {len(eval_dataset)}')


    return train_dataset, eval_dataset