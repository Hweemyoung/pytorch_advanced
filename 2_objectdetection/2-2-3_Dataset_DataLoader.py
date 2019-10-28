# パッケージのimport
import os.path as osp
import random
# XMLをファイルやテキストから読み込んだり、加工したり、保存したりするためのライブラリ
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# VOC2012 Dataset
class VOCDataset(data.dataset):
    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        '''
        :param img_list: list
        :param anno_list: list
        :param phase: str
        :param transform: instance of class DataTransform
        :param transform_anno: instance of class Anno_xml2list
        '''
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        #self.transform = transform
        self.transform = DataTransform(input_size, color_mean)
        #self.transform_anno = transform_anno
        self.transform_anno = Anno_xml2list(voc_classes)

    def __len__(self):
        '''
        :return: length of self.img_list
        '''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        get tensor data of preprocessed img and annotation
        :param index:
        :return:
        '''
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        # 1. load img
        # 1.1 file path of index
        image_file_path = self.img_list[index]
        # 1.2 get [H],[W],[BGR]
        img = cv2.imread(image_file_path)
        # 1.3 get size of img
        height, width, channels = img.shape

        # 2. xml annotations to list
        # 2.1 anno file path of index
        anno_file_path = self.anno_list[index]
        # 2.2 get list of annotations
        anno_list = self.transform_anno(anno_file_path, width, height)

        # 3. preprocess
        # 3.1 transform
        img, boxes, labels = self.transform(img, self.phase, boxes=anno_list[:, :4], labels=anno_list[:, 4])
        # 3.2.1 BGR to RGB
        # 3.2.2 (height, width, RGB) to (RGB, height, width)
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
        # 3.3 pair(BBox, label)
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width

class DataTransform():
    def __init__(self, input_size, color_mean):
        '''
        :param input_size: int
            size to which image will be resized
        :param color_mean: tuple
            (B, G, R). every mean of B, G, R
        '''
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                Expand(color_mean),
                RandomSampleCrop(),
                ToPercentCoords(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ]),
            'val': Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        '''
        :param img:
        :param phase: str
            'train' or 'val'. Mode for preprocessing
        :param boxes:
        :param labels:
        :return:
        '''
        return self.data_transform[phase](img, boxes, labels)

class Anno_xml2list(object):
    def __init__(self, classes):
        '''
        :param classes: list
        '''
        self.classes = classes
    def __call__(self, xml_path, width, height):
        '''
        :param xml_path: str
            path of xml file for specific img
        :param width: int
        :param height: int
        :return:list
            [[xmin, ymin, xmax, ymax, label_ind], ...]
        '''
        ret = []

        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            bndbox = []

            name = obj.find('name').text.lower().strip()
            bbox = obj.fine('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in pts:
                # origin = (1, 1) - 1
                cur_pixel = int(bbox.find(pt).text) - 1
                # normalize
                if pt == 'xmin' or pt == 'xmax':
                    cur_pixel /= width
                else:
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            ret += [bndbox]

        return np.array(ret)

def make_datapath_list(rootpath):
    '''
    :param rootpath: str
    :return ret: train_img_list, train_anno_list, val_img_list, val_anno_list
    '''
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')

    train_id_names = osp.join(rootpath, 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath, 'ImageSets/Main/val.txt')

    train_img_list = []
    train_anno_list = []

    for line in open(train_id_names):
        file_id = line.strip()
        img_path = imgpath_template % file_id
        anno_path = annopath_template % file_id
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    val_img_list = []
    val_anno_list = []

    for line in open(val_id_names):
        file_id = line.strip()
        img_path = imgpath_template % file_id
        anno_path = annopath_template % file_id
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list

def od_collate_fn(batch):
    '''
    customize collate function(minibatch manager)
    :param batch:
    :return imgs: 4-d Tensor
            targets: list
    '''
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0]) # sample[0]: img
        targets.append(torch.FloatTensor(sample[1])) # sample[1]: annotation gt

    # len(imgs) == minibatch size
    # comprises torch.Size([3, 300, 300]) (RGB, width, height)
    # let's convert to torch.Size([batch_num, 3, 300, 300])
    imgs = torch.stack(imgs, dim=0)

    return imgs, targets

rootpath = './data/VOCdevkit/VOC2012'
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

print(train_img_list[0])

voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

transform_anno = Anno_xml2list(voc_classes)

ind = 1
image_file_path = train_img_list[ind]
img =cv2.imread(image_file_path) # [h][w][BGR]
height, width, channels = img.shape

# check operations
transform_anno(val_anno_list[ind], width, height)

# 1. Load image
image_file_path = train_img_list
img = cv2.imread(image_file_path) #[H][W][BGR]
height, width, channels = img.shape

# 2. annotations to list
transform_anno = Anno_xml2list(voc_classes)
anno_list = transform_anno(train_anno_list, width, height)

# 3. show original img
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# 4. create preprocessor class
color_mean = (104, 117, 123)
input_size = 300 # resize to (300, 300)
transform = DataTransform(input_size, color_mean)

# 5. show training img
phase = 'train'
img_transformed, boxes, labels = transform(img, phase, boxes=anno_list[:, :4], labels=anno_list[:, 4])
plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
plt.show()

# set up dataset
# VOC2012 dataset

# checkpoint
color_mean = (104, 117, 123)# BGR mean
input_size = 300# 300x300

train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train', transform=DataTransform(input_size, color_mean), transform_anno=transform_anno)
val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val', transform=DataTransform(input_size, color_mean), transform_anno=transform_anno)

val_dataset.__getitem__(1)

# create dataloader with od_collate_fn
batch_size = 4
train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=od_collate_fn)
val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=True, collate_fn=od_collate_fn)

dataloaders_dict = {
    'train': train_dataloader,
    'val': val_dataloader
}

batch_iterator = iter(dataloaders_dict['val']) #__DataLoaderIter class
images, targets = next(batch_iterator) # return batch
print('images.size == ', images.size())
print('len(images) == ', len(images))
print('target[1].size ==', targets[1].size())

# count number of data in each dataset
print('train_dataset.__len__() == ', train_dataset.__len__())
print('val_dataset.__len__() == ', val_dataset.__len__())