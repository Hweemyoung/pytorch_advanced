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

# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

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

transform_anno(val_anno_list[ind], width, height)