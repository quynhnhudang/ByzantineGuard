import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.attack import soft_label_adversarial_images, FGSM

def test_img(net_g, datatest, args, criterion):
    net_g.eval()
    
    # testing
    test_loss = 0
    correct = 0
    adv_test_loss = 0
    adv_correct = 0
    adv_correct_2 = 0
    misclassified = 0 #change decision
    
    #visualize
    adv_noise =0
    y_preds = []
    y_preds_adv = []
    test_images = []
    test_label = []
    adverserial_images = []
    
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
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
        
        #FGSM method
        (_), images_adv_list = FGSM(args, net_g, criterion, data, target,
                                                   8/255, 0, 1) #The adversarial_images list contains a series of perturbed images
        
        images_adv = torch.stack(images_adv_list) #return a tensor of size (args.bs, 3, 32, 32)
        images_adv = images_adv.to(args.device)
        
        with torch.no_grad():
            
            log_probs = net_g(images_adv)
            # sum up batch loss
            # adv_test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            
            # get the index of the max log-probability
            adv_y_pred = log_probs.data.max(1, keepdim=True)[1]
            adv_correct += (adv_y_pred == 10).int().cpu().sum()
            adv_correct_2 += adv_y_pred.eq(target.data.view_as(adv_y_pred)).long().cpu().sum()
            
        misclassified += (y_pred != adv_y_pred).sum().item()
        y_preds.extend(y_pred.cpu().data.numpy())
        y_preds_adv.extend(adv_y_pred.cpu().data.numpy())
        test_images.extend(data.cpu().data.numpy())
        test_label.extend(target.cpu().data.numpy())
        adverserial_images.extend((images_adv).cpu().data.numpy())

    #test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    #adv_test_loss /= len(data_loader.dataset)
    adv_acc = 100.00 * adv_correct / len(data_loader.dataset)
    
    adv_acc_2 = 100.00 * adv_correct_2 / len(data_loader.dataset)
    
    #visualize
    # visualize_adversarial_images(args, 0, adverserial_images, y_preds, y_preds_adv, 
    #                              test_images, test_label, args.adv_eps)

    if args.verbose:
        print('\nTest set: \nAccuracy: {}/{} ({:.2f}%)\n'.format(
                correct, len(data_loader.dataset), accuracy))
        print('\nAdversarial Test set as a Detector: \nAdv Accuracy: {}/{} ({:.2f}%)\n'.format(
                adv_correct, len(data_loader.dataset), adv_acc))
        
        print('\nAdversarial Test set as a classifier: \nAdv Accuracy: {}/{} ({:.2f}%)\n'.format(
                adv_correct_2, len(data_loader.dataset), adv_acc_2))
        
        print('\nNumber of correct classified clean examples(as compared to clean predictions): {}/{}'.format(correct, len(data_loader.dataset)))
        print('\nNumber of correct classified adversarial examples(as compared to clean predictions): {}/{}'.format(adv_correct, len(data_loader.dataset)))

        print('\nNumber of misclassified adversarial examples: {}/{}'.format(misclassified, len(data_loader.dataset)))

    return accuracy, adv_acc, adv_acc_2

    


