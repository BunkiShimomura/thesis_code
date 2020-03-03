from create_dataset import MyDataset, MyNormalize, divide_dataset
from model import Net, train, test, learn, evaluate

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from skimage import io
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

LABEL_IDX = 0
IMG_IDX = 1

csv_file_path = "/Users/Bunki/Desktop/PMP/code/pytorch_tutorial/script/classifier/data.csv"
ROOT_DIR = "/Users/Bunki/Desktop/PMP/code/pytorch_tutorial"

imgDataset = MyDataset(csv_file_path, ROOT_DIR, transform=transforms.Compose([
    transforms.Resize(50),
    transforms.ToTensor(),
    MyNormalize()
    ]))

train_loader, test_loader, validation_loader = divide_dataset(imgDataset, 0.2, 16, 16)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

epoch = [1, 3]
train_results = []
test_results = []
for ep in epoch:
    train_result, test_result, epoch = learn(train_loader, test_loader, ep)
    train_results.append([train_result, epoch])
    loss, accuracy = test_result
    print(loss)
    print(accuracy)
    print(test_result)
    print(epoch)
    test_results.append([loss, accuracy, epoch])

print(train_results)

for result in train_results:
    print('epoch: ' + str(result[1]) + ", " + "Average loss: " + str(result[0]))

for result in test_results:
    print('epoch: ' + str(result[2]) + ", " + "Average loss: " + str(result[0]) + ", " + "Accuracy: " + str(result[1]))

print("evaluate")
evaluate(validation_loader, 'cnn_dict.model')
