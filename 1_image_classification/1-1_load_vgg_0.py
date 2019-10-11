# パッケージのimport
import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import models, transforms

class BaseTransform():
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose(
            [transforms.Resize(resize),
             transforms.CenterCrop(resize),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)]
        )

    def __call__(self, img):
        return self.base_transform(img) #transforms.Compose.__call__(*arg) returns transformed *arg

class ILSVRCPredictor():
    def __init__(self, class_index): # type(class_index) = dict
        self.class_index = class_index

    def predict_max(self, out):
        '''
        :param out: torch.Size([1, 1000])
            Output from Net
        :return: str
            Predicted label name
        '''
        maxid = np.argmax(out.detach().numpy) #torch.Tensor.detach: Returns a new Tensor, detached from the current graph. The result will never require gradient.
        predicted_label_name = self.class_index[str(maxid)][1]

        return predicted_label_name

# PyTorchのバージョン確認
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.eval()

base_dir = os.getcwd()
image_file_path = os.path.join(base_dir, 'data/goldenretriever-3724972_640.jpg')
img = Image.open(image_file_path)

plt.imshow(img)
plt.show()

resize = 224
mean = (.485, .456, .406)
std = (.229, .224, .225)
transform = BaseTransform(resize, mean, std)
img_transformed = transform(img)

img_transformed = img_transformed.numpy().transpose((1, 2, 0)) # (color, height, width) -> (height, width, color)
img_transformed = np.clip(img_transformed, 0, 1) # clip range = [0, 1]

plt.imshow(img_transformed)
plt.show()

base_dir = os.getcwd()
ILSVRC_class_index = json.load(open(os.path.join(base_dir, '/data/imagenet_class_index.json'), 'r'))

predictor = ILSVRCPredictor(ILSVRC_class_index)

inputs = img_transformed.unsqueeze_(0) # torch.Size([1, 3, 224, 224]) # torch.Tensor.unsqueeze_: In-place version of :meth:`~Tensor.unsqueeze`

out = net(inputs)
result = predictor.predict_max(out)

print("predicted result:", out)