import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torch.nn.utils import clip_grad_norm_

import numpy as np
from random import randint
import math

from src.test import test_img
from src.attack import soft_label_adversarial_images, FGSM

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]][0], self.dataset[self.idxs[item]][1]
        label = torch.Tensor(label)
        # print(f'item, {item} label {label}')
        return image, label

class FGSM_ModelUpdate(object):
    def __init__(self, args, client_id, local_ep=1, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.local_ep = local_ep
        self.client_id = client_id

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
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                images_list, adversarial_images = FGSM(self.args, net, self.loss_func,                                                   images, labels, \
                                                    epsilon = 8/255, 
                                                    min_val = 0, 
                                                    max_val = 1)
                    
                soft_label_list = soft_label_adversarial_images(self.args, adversarial_images, 
                                 self.args.sl_tau_FGSM, 
                                self.args.sl_alpha_FGSM,
                                images_list, labels)

                adversarial_images = torch.stack(adversarial_images).to(self.args.device) #convert list to tensor
                soft_label_list = [torch.tensor(label) for label in soft_label_list] #convert each element to a tensor
                soft_label_list = torch.stack(soft_label_list).to(self.args.device) #conver list to tensor

                net.zero_grad()
                log_probs = net(torch.cat([images, adversarial_images], dim=0)) #model predictions
                loss = self.loss_func(log_probs, torch.cat([labels, soft_label_list], dim=0)) #labels here 11 class and clean examples
                    
                loss.backward()
                
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Round : {} Party {}: Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        round_cur, pid, iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
           
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)