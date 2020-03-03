#test code to verify various inputs and outputs

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = input("path: ")


img = Image.open(image)
im = np.asarray(img.convert('L'))
print(im)
print(im.size)
print(im.shape)
plt.imshow(img)
plt.show()
