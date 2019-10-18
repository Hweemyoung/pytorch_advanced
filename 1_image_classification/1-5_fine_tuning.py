import glob
import os
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

from make_dataset_dataloader import ImageTransform, make_datapath_list, HymenopteraDataset

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    # 1. set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    # 2. network to device
    net.to(device)
    # 3. loop over epoch
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch, num_epochs))
        print('--------------')
        # 4. iterate upon phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            # 5. initialize loss per phase
            epoch_loss = .0
            epoch_correct = 0

            # 6. skip first training phase
            if phase == 'train' and epoch == 0:
                continue # to phase'val'

            # 7. iterate dataloader
            for inputs, labels in tqdm(dataloaders_dict[phase]): #dataloader는 자체로 iterable
                # 8. dataset to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 9. initialize grad
                optimizer.zero_grad()

                # 10. forward
                with torch.set_grad_enabled(mode=(phase == 'train')): #enable grad only when training # with + context_manager
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, dim=1) # returns values, indices

                    # 11. (training)calc grad
                    if phase == 'train':
                        loss.backward()
                        # 12. update parameters
                        optimizer.step()

                    # 13. add loss and correct per minibatch per phase
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_correct += (preds == labels.data).sum() ## labels.data?

            # 14. print epoch summary
            epoch_loss /= len(dataloaders_dict[phase].dataset) ## len(dataloader): num of datum
            epoch_acc = epoch_correct.double() / len(dataloaders_dict[phase].dataset)

            print('Epoch loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))






train_list = make_datapath_list(phase='train')
val_list = make_datapath_list(phase='val')

size = 224
mean = (.485, .456, .406)
std = (.229, .224, .225)

# Dataset
train_dataset = HymenopteraDataset(file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')
val_dataset = HymenopteraDataset(file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')

# DataLoader
batch_size = 32
train_dataloader = data.DataLoader(
    train_dataset, batch_size, shuffle=True
)
val_dataloader = data.DataLoader(
    val_dataset, batch_size, shuffle=True
)

dataloaders_dict = {
    'train': train_dataloader,
    'val': val_dataloader
}

use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.classifier[6] = nn.Linear(4096, 2)

net.train()
print('Network set-up completed. Loaded pre-trained weights, set to training mode.')

criterion = nn.CrossEntropyLoss()

params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

update_param_names_1 = ['features']
update_param_names_2 = ['classifier.0.weight',
                        'classifier.0.bias',
                        'classifier.3.weight',
                        'classifier.3.bias']
update_param_names_3 = ['classifier.6.weight',
                        'classifier.6.bias']

for name, param in net.named_parameters():
    if update_param_names_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
        print('loaded on params_to_update_1:', name)

    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        print('loaded on params_to_update_2:', name)

    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        print('loaded on params_to_update_3:', name)

    else:
        param.requires_grad = False
        print('no_grad:', name)

optimizer = optim.SGD([
    {'params': params_to_update_1, 'lr': 1e-4},
    {'params': params_to_update_2, 'lr': 5e-4},
    {'params': params_to_update_3, 'lr': 1e-3}], momentum=.9)

num_epochs = 2
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs)

# save net parameters
save_path = osp.join(os.getcwd(), 'weights_fine_tuning.pth')
torch.save(net.state_dict, save_path) ## net.state_dict?

# load net parameters
load_path = save_path
load_weights = torch.load(load_path, map_location={'cuda0': 'cpu'})
net.load_state_dict(load_weights)


