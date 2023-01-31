# IMPORTS
# ------------------------
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random 
# ------------------------

# CONSTANTS
# ------------------------
BATCH_SIZE = 64
CLASSES = [0, 1]
NOVEL_CLASS = [2]
ALL_CLASSES = CLASSES + NOVEL_CLASS
PERC_VAL = 0.20
# ------------------------

# func: train_ind( [] ) => []
# return indices matching non-novel classes
def train_ind(dataset):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in CLASSES:
            indices.append(i)

    return indices


# func: test_ind() => []
# return indices matching non-novel classes 
#                         + novel classes
def test_ind(dataset, val=False):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in ALL_CLASSES:
            indices.append(i)
    
    # if this is the validation set, return PERC_VAL% of the data      
    if val == True: return indices[:int(PERC_VAL * len(indices))]
    else: return indices[int(PERC_VAL * len(indices)):]


# Normalize input data...

trainset = torchvision.datasets.MNIST(root='./data', download=True, 
                                     transform=torchvision.transforms.ToTensor())

train_inds = train_ind(trainset)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, sampler = SubsetRandomSampler(train_inds))


testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, 
                                     transform=torchvision.transforms.ToTensor())
val_inds = test_ind(testset, val=True)
val_loader = DataLoader(testset, batch_size=BATCH_SIZE, sampler = SubsetRandomSampler(val_inds))

test_inds = test_ind(testset)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE,
                                         sampler = SubsetRandomSampler(test_inds))

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# FUNC: imshow( [] ) => None
# SUMMARY: visualize the sampels
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
print(images[0].shape)
# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
labels = np.array(labels)
df = pd.DataFrame(labels.reshape( (int(np.sqrt(BATCH_SIZE)), int(np.sqrt(BATCH_SIZE)) )))
print(df)

# util functions
# ---------------
def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def random_weight(shape):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    """
    if len(shape) == 2:  # FC weight
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:]) # conv weight [out_channel, in_channel, kH, kW]
    # randn is standard normal distribution generator. 
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w

def zero_weight(shape):
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)


USE_GPU = True
dtype = torch.float32 # we will be using float throughout this tutorial
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# create a weight of shape [3 x 5]
# you should see the type `torch.cuda.FloatTensor` if you use GPU. 
# Otherwise it should be `torch.FloatTensor`
random_weight((3, 5))

# Constant to control how frequently we print train loss
print_every = 100
print('using device:', device)


channel_1 = 8
channel_2 = 8
channel_3 = 8
channel_4 = len(CLASSES)

input_dims = 784 # 1 channels x 28 by 28 images

# FOR DEBUGGING
from NovelNetwork import NovelNetwork
from collections import OrderedDict

layers = OrderedDict([
          ('flatten', nn.Flatten()),
          ('lin1', nn.Linear(784, channel_1)),
          ('relu1', nn.ReLU()),
          ('lin2', nn.Linear(channel_1, channel_2)),
          ('relu2', nn.ReLU()),
          ('lin3', nn.Linear(channel_2, channel_3)),
          ('relu3', nn.ReLU()),
          ('lin4', nn.Linear(channel_3, channel_4)),
          ('relu4', nn.ReLU()),
          ('softmax', nn.Softmax(dim=1)),
        ])

args = {
  'print_every' : 100,
  'feat_layer'  : 'softmax',
  'feat_sample' : 50,
  'min_g' : 3,
  'max_g' : 10,
  'epoch' : 5,
  'lr' : 5e-3,
  'dist_metric' : 'mahalanobis'
}
new_model = NovelNetwork(layers, known_labels=CLASSES)
new_model.train(train_loader, val_loader, args)

acc, raw_acc, info = new_model.test_analysis(test_loader, print_info=True)
print('Acc: ', acc, ' | Raw Acc: ', raw_acc)
print('conf: ', info['conf']['acc'], ' | delta: ', info['conf']['delta'])

#new_model.plot_feats(test_loader, 'lin1')
#new_model.plot_feats(test_loader, 'lin2')
#new_model.plot_feats(test_loader, 'lin3')
#new_model.plot_feats(test_loader, 'lin4')
#new_model.plot_feats(test_loader, 'softmax')