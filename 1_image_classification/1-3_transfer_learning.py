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

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class ImageTransform():
    """
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(.5, 1)),
                transforms.RandomHorizontalFlip(p=.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        """
        :param phase: 'train' or 'normal'
        """
        return self.data_transform[phase](img)


class HymenopteraDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        '''
        :param file_list: a list
        :param transform: an instance of torchvision.transform class
        :param phase: str
        '''
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''

        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅][色RGB]

        # 画像の前処理を実施
        img_transformed = self.transform(
            img, self.phase)  # torch.Size([3, 224, 224])

        # 画像のラベルをファイル名から抜き出す
        label = img_path.split('/')[-2]

        # ラベルを数値に変更する
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1

        return img_transformed, label


def make_datapath_list(phase='train'):
    """
    :param phase:
    :return:
    path_list : list
    """
    rootpath = osp.join(os.getcwd(), 'data/hymenoptera_data')
    target_path = osp.join(rootpath, phase, '**/*.jpg')
    print('target_path:', target_path)

    path_list = [path for path in glob.glob(target_path)]

    return path_list

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('----------')
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train() # switch the model to training mode
            else:
                net.eval() # switch the model to eval mode

            epoch_loss = .0 # loss-sum of each epoch
            epoch_corrects = 0 # number of correct predictions

            # 학습 전의 성능 확인을 위해 for 루프로 되돌려 보낸다
            if (epoch == 0) and (phase == 'train'):
                continue

            # get mini-batch from dataloader
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                # initialize gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(mode=(phase == 'train')): # torch.set_grad_enabled? # set grad true when training
                    outputs = net(inputs)
                    loss = criterion(outputs, labels) # loss = nn.loss()(outputs, labels)
                    _, preds = torch.max(outputs, dim=1) # predict label # returns values, indices

                    # back_prop if training
                    if phase == 'train':
                        # calc grads
                        loss.backward()
                        # update parameters with designated optimizer
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0) # loss * batch_size
                    epoch_corrects += (preds == labels).sum()

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset) #len(torch.utils.data.DataLoader.dataset) # returns len(dataset.file_list)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


base_dir = os.getcwd()
image_file_path = osp.join(base_dir, 'data/goldenretriever-3724972_640.jpg')
img = Image.open(image_file_path)  # [height][width][RGB]

plt.imshow(img)
plt.show()

size = 224
mean = (.485, .456, .406)
std = (.229, .224, .225)

transform = ImageTransform(size, mean, std)
img_transformed = transform(img, phase='train')
img_transformed.size

img_transformed = img_transformed.numpy().transpose((1, 2, 0))
img_transformed = np.clip(img_transformed, 0, 1)
print(img_transformed.shape)

plt.imshow(img_transformed)
plt.show()

train_list = make_datapath_list(phase='train')
val_list = make_datapath_list(phase='val')

print(train_list)

train_dataset = HymenopteraDataset(file_list=train_list, transform=transform, phase='train')
val_dataset = HymenopteraDataset(file_list=val_list, transform=transform, phase='val')

index = 0
print('The size of the first training image is', train_dataset.__getitem__(index)[0].size())
print(train_dataset.__getitem__(index)[1])

batch_size = 32
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True) # data.DataLoader(dataset=data.Dataset, batch_size=int, shuffle=bool)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=True)

dataloaders_dict = {
    "train": train_dataloader,
    "val": val_dataloader}

batch_iterator = iter(dataloaders_dict["train"]) # convert dataloader to an iterator
inputs, labels = next(batch_iterator) # pick up the first element
print(inputs.size())
print(labels)

use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)

net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

net.train()

print('Network set-up completed: Loaded pre-trained weights, switched to training mode.')

criterion = nn.CrossEntropyLoss()

params_to_update = []

update_param_names = ["classifier.6.weight", 'classifier.6.bias']

# 새로 생성한 Layer-6만 grad 허용
for name, param in net.named_parameters():
    if name in update_param_names:
        param.required_grad = True
        params_to_update.append(param)
    else:
        param.required_grad = False

print("----------")
print(params_to_update)

optimizer = optim.SGD(params=params_to_update, lr=.001, momentum=.9) # optim.SGD(params: list of tensors with grad)

num_epochs = 2
train_model(net=net,
            dataloaders_dict=dataloaders_dict,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs)