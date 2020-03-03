from create_dataset import MyDataset, MyNormalize, divide_dataset
from model import Net, train, test, learn

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

csv_file_path = "/Users/Bunki/Desktop/PMP/pytorch_tutorial/script/classifier/data.csv"
ROOT_DIR = "/Users/Bunki/Desktop/PMP/pytorch_tutorial"

#print(pd.read_csv(csv_file_path))

imgDataset = MyDataset(csv_file_path, ROOT_DIR, transform=transforms.Compose([
    transforms.Resize(50),
    transforms.ToTensor(),
    MyNormalize()
    ]))

train_loader, test_loader = divide_dataset(imgDataset, 0.2, 16, 16)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

for epoch in range(2):
    train(epoch, train_loader)
torch.save(model.state_dict(), 'cnn_dict.model')
torch.save(model, 'cnn.model')



import pandas as pd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# Sequential() contains modules in the oder that they are passed in the constructor
# feature_fn contains the part of model which extracts features
feature_fn = torch.nn.Sequential(*list(model.children())[:-2]).cpu()
# classifier_fn contains the part of model which classifies images
classifier_fn = torch.nn.Sequential(*(list(model.children())[-2:-1] + [Flatten()] + list(model.children())[-1:]))

def GradCam(img, c, feature_fn, classifier_fn):
    feats = feature_fn(img.cpu())
    print(feats.size())
    _, N, H, W = feats.size()
    print(_, N, H, W)
    out = classifier_fn(feats.view(feats.size(0), -1))
    c_score = out[0, c]
    print(c_score)
    grads = torch.autograd.grad(c_score, feats)
    w = grads[0][0].mean(-1).mean(-1)
    sal = torch.matmul(w, feats.view(N, H*W))
    sal = F.relu(sal)
    # view() resizes the tensor
    sal = sal.view(H, W).cpu().detach().numpy()
    # matmul() excutes multiplization
    sal = np.maximum(sal, 0)
    return sal

# obtain a single image from dataset
input_index = 15
input_data = test_loader.dataset[input_index][0]
print(input_data)
print(type(input_data))
input_data = input_data.view(1, input_data.shape[0], input_data.shape[1], input_data.shape[2]).cpu()

# obtain output and label of top-rated class
pp, cc = torch.topk(nn.Softmax(dim=1)(model(input_data)), 1)

# obtain saliency map
sal = GradCam(input_data.cpu(), cc[0][0], feature_fn, classifier_fn)

# visualize saliency map
img = input_data.permute(0, 2, 3, 1).view(input_Data.shape[2], input_data.shape[3], input_data.shape[1]).cpu().numpy()
img_sal = Image.fromarray(sal).resize(img.shape[0:2], resample=Image.LINEAR)

plt.imshow(img)
plt.imshow(np.array(img_sal), alpha=0.5, cmap='jet')
plt.colorbar()
