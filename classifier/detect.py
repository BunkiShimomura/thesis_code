# Refference
# http://cedro3.com/ai/pytorch-ssd/

import os
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

from model import Net

# Load model
model = Net()
param = torch.load('cnn_dict.model')
model.load_state_dict(param)

# Define function to detect designated features
def detect(image, count):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    x = cv2.resize(gray_image, (50, 50)).astype(np.float32)
    # x -= (104.0, 117.0, 123.0) # Make the color grayscale by substracting RGB mean
    x = x.astype(np.float32)
    print(x.shape)
    x = x[:, ::-1].copy()
    # x = torch.from_numpy(x).permute(2, 0, 1)
    x = torch.from_numpy(x)
    xx = Variable(x.unsqueeze(0))
    xx = Variable(xx.unsqueeze(0))
    print(xx.shape)
    y = model(xx)

    # from data import VOC_CLASSES as labels
    plt.figure(figsize=(10, 6))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)
    currentAxis = plt.gca()
    detections = y.data
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)

    for i in range(detections.size(1)):
        j = 0
        print(detections)
        print(detections.shape)
        print(type(detections))
        while detections[0, i, j, 0] <= 0.6:
            score = detections[0, i, j, 0]
            # label_name = labels[i-1]
            label_name = str(i)
            display_txt = '%s: %.2f'%(label_name, score)
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, lineiwidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            j += 1
        plt.savefig('detext_img/' + '{0.04d}'.format(count) + '.jpg')
        plt.close()

def main():
    files = sorted(glob.glob('image_dir/*.jpg'))
    print(files)
    count = 1
    for i, file in enumerate(files):
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        detect(image, count)
        print(count)
        count += 1

if __name__ == '__main__':
    main()
