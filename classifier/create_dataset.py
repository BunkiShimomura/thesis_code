#Reference
#https://qiita.com/sheep96/items/0c2c8216d566f58882aa

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision as tv
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from skimage import io
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import cv2

#Generate dataset from csv

LABEL_IDX = 0
IMG_IDX = 1

class MyDataset(Dataset):

    def __init__(self, csv_file_path, root_dir, transform=None):
        self.image_dataframe = pd.read_csv(csv_file_path)
        self.root_dir = root_dir
        #画像データの処理
        self.transform = transform

    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        label = self.image_dataframe.iat[idx, LABEL_IDX]
        img_name = os.path.join(self.root_dir, 'data', 'load_data', self.image_dataframe.iat[idx, IMG_IDX]) #画像ファイルへのPATHを作る
        #画像の読み込み
        image = cv2.imread(img_name)
        #画像への処理
        if self.transform:
            im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im2 = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            image = Image.fromarray(im2)
            image = self.transform(image)

        return image, label


class MyNormalize:
    def __call__(self, image):

        return image


#Generate testdata and traindata using Dataset
def divide_dataset(dataset, test_size, batch_size_train, batch_size_test):
    learning_data, validation_data = train_test_split(dataset, test_size=0.05)
    train_data, test_data = train_test_split(learning_data, test_size=0.2)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader, validation_loader
