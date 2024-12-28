import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torch.nn.utils import clip_grad_norm_

import numpy as np
from random import randint
from collections import OrderedDict

from src.test import test_img


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class SCAFFOLD_ModelUpdate(object):
    def __init__(self, args, client_id, local_ep=1, dataset=None, idxs=None):
        self.args = args
        if self.args.model == 'linear_regression':
            print('MSE loss')
            self.loss_func = torch.nn.MSELoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, 
                                    shuffle=True)
        self.local_ep = local_ep
        #self.c_local: Dict[List[torch.Tensor]] = {} #client's control
        self.c_diff = [] # -c_l + c_g, controls differences
        self.client_id = client_id
            
    
    #c_local dictionary is shared with server which is not private.
    #To do: keep c_lcaol dictionary inside the ModelUpdate class object.
    def train(self, pid, round_cur, local_net, net, c_local, c_global):
        
        local_net.train()
        
        # train and update
        # weight_decay=self.args.l2_lambda
        if self.args.l2_lambda != 0:
            optimizer = torch.optim.SGD(local_net.parameters(), lr=self.args.lr, weight_decay=self.args.l2_lambda, momentum=self.args.momentum)
        else:
            optimizer = torch.optim.SGD(local_net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []

        if self.args.sys_homo: 
            local_ep = self.local_ep
        else:
            local_ep = randint(self.args.min_le, self.args.max_le)
        
        # print(f'self.client_id not in c_local.keys() {self.client_id not in c_local.keys()}')
        if self.client_id not in c_local.keys():#first round
            self.c_diff = c_global
        else:
            self.c_diff = [] #reinitialize
            for c_l, c_g in zip(c_local[self.client_id], c_global):
                # print(f'id {self.client_id} c_l {c_l.cpu().detach().numpy()}')
                # print(f'id {self.client_id} c_g {c_g.cpu().detach().numpy()}')
                self.c_diff.append(-c_l + c_g)
                # print(f'id {self.client_id} c_diff = {self.c_diff}')
        
        # if self.client_id in c_local.keys():
        #     print("Client {}: Before training: w = {:4.3f} c_local {}".format(self.client_id, \
        #                                                        local_net.weight.item(), \
        #                                                         c_local[self.client_id], \
        #                                                            c_global))
        # #train     
        for iter in range(local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                local_net.zero_grad()
                log_probs = local_net(images)
                loss = self.loss_func(log_probs, labels)
                        
                loss.backward()
                
                #Compute mini-batch gradient
                for param, c_d in zip(local_net.parameters(), self.c_diff):
                    # print(f'id {self.client_id} {batch_idx} param.grad {param.grad.cpu().detach().numpy()}')
                    # print(f'id {self.client_id} {batch_idx} c_d.data {c_d.data.cpu().detach().numpy()}')
                    param.grad += c_d.data #compute mini-batch gradient g_i(y_i)
                    # print(f'id after {self.client_id} param.grad {param.grad.cpu().detach().numpy()}')
                
                #Compute the new weights via the new mnini-batch gradient
                optimizer.step() #yi←yi−ηl(gi(yi)−ci + c)

                if self.args.verbose:
                    print('Round : {} Party {}: Update Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        round_cur, self.client_id, iter, batch_idx * len(images), 
                        len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        #end epoch for loop
        # compute and update local control variate
        #model_params: OrderedDict[str, torch.Tensor],
        # model_params = local_net.parameters() #need to double check
         
        with torch.no_grad():
            if self.client_id not in c_local.keys(): #first round and c_local maintains a state(related to weight) for each clien
                c_local[self.client_id] = [
                    torch.zeros_like(param, device=self.args.device)
                    for param in local_net.parameters() #local model
                ]

            y_delta = []
            c_plus = []
            c_delta = []

            # compute y_delta (difference of model before and after training)
            for param_l, param_g in zip(local_net.parameters(), net.parameters()):
                y_delta.append(param_l - param_g)

            # compute c_plus
            coef = 1 / (local_ep * self.args.lr) #1/ (K * local_lr)
            for c_l, c_g, diff in zip(c_local[self.client_id], c_global, y_delta):
                c_plus.append(c_l - c_g - coef * diff) #c_plus and diff means y_delta

            # compute c_delta, controls differences
            for c_p, c_l in zip(c_plus, c_local[self.client_id]):
                c_delta.append(c_p - c_l) #c_delta

            c_local[self.client_id] = c_plus #c_l = c_pluts
        
            # print("Client {}: After training: w = {:4.3f} w_delta {}, coef {}, cplus {} c_delta {}".format(self.client_id, local_net.weight.item(), \
            #      y_delta, coef, c_plus, c_delta))
        
            #In here we return local_ndel.state_dict()
            # Difference between parameters() and state_dict()
            # https://stackoverflow.com/questions/54746829/pytorch-whats-the-difference-between-state-dict-and-parameters#:~:text=The%20parameters()%20only%20gives,an%20iterator%20over%20module%20parameters.&text=On%20the%20other%20hand%2C%20state_dict,parameters%20but%20also%20buffers%20%2C%20etc.

        return local_net.state_dict(), sum(epoch_loss) / len(epoch_loss), (y_delta, c_delta), c_plus