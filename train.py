import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import os
import imageio
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from data import *
from torch import multiprocessing, cuda
from torch.backends import cudnn
from unet3D import *

import torchvision.models
from torchvision.utils import save_image
from PIL import Image

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

from torch.utils.data import Subset
import numpy as np
import math

#Set device
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

#Hyperparameters
in_channel = 4
num_classes = 4
lr = 0.001
batch_size = 1
num_epochs = 100
load_model = False

saved_model_path = "checkpoint_v11.pth.tar"

my_transforms = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=(-20, 20), translate=(0.1, 0.1),
                 scale=(0.9, 1.1), shear=(-0.2, 0.2)),
    ElasticTransform(alpha=720, sigma=24),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0],std=[1.0])
])

# dataset = BRATS_Dataset(csv_file = '../MICCAI_BraTS_2019_Data_Training/HGG_CSV.csv', root_dir = '../MICCAI_BraTS_2019_Data_Training/HGG/', transform = my_transforms)

dataset = BRATS_Dataset(csv_file = '../MICCAI_BraTS_2019_Data_Training/HGG_CSV.csv', root_dir = '../MICCAI_BraTS_2019_Data_Training/HGG/', transform = None)


train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


#Initialize Network
net = Unet()
model = net.cuda()



#Saving the model after every 2 epochs
def save_checkpoint(checkpoint,filename = saved_model_path):
    torch.save(checkpoint,filename)

#Load the saved model
def load_checkpoint(checkpoint):
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

if load_model:
    load_checkpoint(torch.load(saved_model_path))

seed = 42

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True

###########   LOSS & OPTIMIZER   ##########
# criterion = nn.CrossEntropyLoss()

weights = torch.Tensor([0.05, 0.95, 0.95, 0.95])
momentum = 0.9
step_size = 15
gamma = 0.5

bce = nn.BCELoss()
criterion = nn.CrossEntropyLoss(ignore_index=4,weight=weights)

optimizer = optim.SGD(model.parameters(),lr=lr, momentum=momentum, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

#Train the network
for epoch in range(1,num_epochs+1):
    # epoch = i+1
    losses = []

    if epoch%5==0:
        checkpoint = {'state_dict':net.state_dict(),'optimizer':optimizer.state_dict()}
        save_checkpoint(checkpoint)
        model.eval()
        # evaluate(val_loader,'valid')
    scheduler.step()
    print('Learning rate= '+str(optimizer.param_groups[0]['lr']))
    for batch_idx, (image, mask) in tqdm(enumerate(train_loader)):
        image = image.cuda()
        image = image.type(dtype)
        mask = mask.cuda()
        mask = mask.type(dtype)

        # image = torch.unsqueeze(image, 0)
        print(image.shape)
        print(mask.shape)

        predicted_mask = model(image)
        print(predicted_mask.shape)