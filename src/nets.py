import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms 
from torchvision import models


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

class CNN_v1(nn.Module):
    def __init__(self, args):
        super(CNN_v1, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes + 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_v2(nn.Module):
    def __init__(self, args):
        super(CNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes + 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_v3(nn.Module):
    def __init__(self, args):
        super(CNN_v3, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)

        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, args.num_classes + 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Alexnet(nn.Module):

    def __init__(self, args):
        super(Alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(args.num_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #######################################
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #######################################
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #######################################
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #######################################
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #######################################
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096), #256 * 6 * 6 = 9216
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(1024, args.num_classes + 1),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_model(name="vgg16", pretrained=False):
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif name == "resnet50":
        model = models.resnet50(pretrained=pretrained)  
    elif name == "densenet121":
        model = models.densenet121(pretrained=pretrained)       
    # elif name == "alexnet":
    #     model = models.alexnet(pretrained=pretrained) #error
    elif name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
    elif name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
    # elif name == "inception_v3":
    #     model = models.inception_v3(pretrained=pretrained) #error
    elif name == "googlenet":       
        model = models.googlenet(pretrained=pretrained)
    return model

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
    elif args.model == 'vgg16' or args.model == 'vgg19':
        net_glob = get_model(args.model)
        for param in net_glob.parameters():
            param.requires_grad = False #fixed feature extractor
        num_ftrs = net_glob.classifier[6].in_features
        net_glob.classifier[6] = nn.Linear(num_ftrs, args.num_classes + 1)
        net_glob.to(args.device)
    elif args.model == 'densenet121':
        net_glob = get_model(args.model)
        for param in net_glob.parameters():
            param.requires_grad = True #fixed feature extractor
        num_ftrs = net_glob.classifier.in_features
        net_glob.classifier = nn.Linear(num_ftrs, args.num_classes + 1)
        net_glob.to('cuda') 
    else:
        #Resnet or GoogleNet
        net_glob = get_model(args.model)
        for param in net_glob.parameters():
            param.requires_grad = True #fixed feature extractor
        num_ftrs = net_glob.fc.in_features
        net_glob.fc = nn.Linear(num_ftrs, args.num_classes + 1)
        net_glob.to(args.device)
        
    return net_glob