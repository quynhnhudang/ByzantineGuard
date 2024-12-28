import numpy as np
from torchvision import datasets, transforms

def iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: a dictionary where the keys are the user indices 
    and the values are sets of indices of images in the dataset that belong to that user.
    """
    num_items = int(len(dataset)/num_users)
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    for i in range(num_users):
        
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) #sampling without replacement
        all_idxs = list(set(all_idxs) - dict_users[i]) #rmoves selected indices from all_idxs.
    
    return dict_users

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
    
    min_num = 100
    max_num = 700

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
    if args.sampling == 'one-class-noniid':
        num_shards = args.num_users #4 
    elif args.sampling == 'two-class-noniid':
        num_shards = args.num_users * 2
    else:
        exit('Error. Non define non-iid sampling method.')

    num_imgs = int(num_dataset/num_shards) #10000 # [50,000 - (classwise * 10)  training imgs]/share -->  40,000 imgs/4 then each shard X 10,000 images
    
    idx_shard = [i for i in range(num_shards)] #[0, 1, 2, 3]
    dict_users = {i: list() for i in range(num_users)} #['0': np.array[], '1': np.array[], ... ]
    
    idxs = np.arange(num_shards*num_imgs) #[0, 1, 2, 3,..., 39999]

    # labels = dataset.train_labels.numpy()
    if args.dataset == "mnist":
        labels = dataset.targets.numpy()
    elif args.dataset == "cifar" or args.dataset == "adv_cifar":
        labels = np.array(dataset.targets) #an array of size 40,000 with class labels from 0 to 9.
    else:
        exit('Error: unrecognized dataset')

    # sort labels again. That's why it might be not necessary
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    
    idxs = idxs_labels[0]
    labels = idxs_labels[1]

    #for each user
    
    if args.sampling == 'one-class-noniid':
        for i in range(num_users):  
            rand = np.random.choice(idx_shard, 1, replace=False) # rand is a np.array has only 1 element.
            idx_shard = list(set(idx_shard) - set(rand)) #update idx_shard
            rand = rand.item() #convert np.array to integer
            rand_set = idxs[rand*num_imgs:(rand+1)*num_imgs]
            #divide the idxs list into partitions, and assign [rand*num_imgs, (rand+1)*num_imgs] to the user
            dict_users[i] = rand_set         

    elif args.sampling == 'two-class-noniid':
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                rand_shard = idxs[rand*num_imgs:(rand+1)*num_imgs]
                print(f'rand {rand}, rand_share {len(rand_shard)}')
                dict_users[i].append(rand_shard)
   
    else:
        exit('Error. Non define non-iid sampling method.')

    #Class dist per party
    if args.sampling == 'one-class-noniid':
        for idx in range(num_users):
            # Use indices for train/test subsets
            train_indices = dict_users[idx]
            y_train_pi = list()
            for i in list(train_indices):
                y_train_pi.append(dataset[i][1])
            
            y_train_pi = np.array(y_train_pi)
            print_statistics(idx, y_train_pi, 10)
    elif args.sampling == 'two-class-noniid':
        for idx in range(num_users):
            # Use indices for train/test subsets
            train_indices = dict_users[idx]
            y_train_pi = list()
            for i in range(len(train_indices)):
                for j in train_indices[i]:
                    y_train_pi.append(dataset[j][1])
            y_train_pi = np.array(y_train_pi)
            print_statistics(idx, y_train_pi, 10)
    else:
         exit('Error. Non define non-iid sampling method.')

    return dict_users



