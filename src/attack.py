import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
import math

def soft_label_adversarial_images(args, adversarial_images, sl_tau, sl_alpha, 
                                images_list, label_list):
    
    adversarial_images = torch.stack(adversarial_images).cpu()
    # print(adversarial_images.size())
    images_list = torch.stack(images_list).cpu().detach().numpy() #detach
    label_list = label_list.cpu()
    
    adversarial_images = np.array(adversarial_images)
    images_list = np.array(images_list)
    label_list = np.array(label_list)
    
    label_list_soft = np.copy(label_list)

    perturbation = np.zeros(len(label_list)) # record the size of perturbation (2-norm)
    for i in range(len(label_list)):
        diff = images_list[i]-adversarial_images[i]
        
        #L2-norm
        if args.dataset == 'cifar':
            diff = diff.reshape((32*32,3))
            perturbation[i] = LA.norm(diff)/(32*math.sqrt(3))
        else:
            diff = diff.reshape((28*28,1))
            perturbation[i] = LA.norm(diff)/(28)
        #For each image
        true_label = np.argmax(label_list_soft[i])
        if perturbation[i] < sl_tau:
            one_hot = [0] * (args.num_classes + 1)
            one_hot[true_label] = sl_alpha
            one_hot = [1/(args.num_classes)*(1 - sl_alpha) if j != true_label else val for j, val in enumerate(one_hot)]
            # print(f'label_list_soft[i] {label_list_soft[i].shape}')
            # print(f'one_hot {one_hot}')
            label_list_soft[i] = np.array(one_hot)
        else:
            beta = sl_alpha/2 + (1-sl_alpha)/11
            gamma = beta
            one_hot = [0] * (args.num_classes + 1)
            one_hot[true_label] = beta
            one_hot = [1/(args.num_classes - 1)*(1 - sl_alpha) if j != true_label else val for j, val in enumerate(one_hot)]
            one_hot[-1] = gamma
            # label_list_soft[i, :] = (1/9)*(1 - beta - gamma) #evenly distributes the remaining probability mass among the other classes.
            label_list_soft[i] = np.array(one_hot)
        
    return label_list_soft.tolist()


def visualize_adversarial_images(args, round_cur, adversarial_images, y_preds, y_preds_adv, images_list, label_list, epsilon):
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
        
        name_file = f'./save/{args.fed}_{args.adv_eps}_{args.dataset}_{args.model}_{args.local_ep}_nParties_{len(idxs_users)}_{args.sampling}_{args.classwise}_{args.alpha}_round_cur_{round_cur}'
        name_file_1 = name_file + '_visualization.pdf'
        plt.savefig(name_file_1)
        plt.close()
        
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