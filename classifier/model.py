import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torch.autograd import Variable
from sklearn.metrics import classification_report
from PIL import Image
import matplotlib.pyplot as plt
from visdom import Visdom
import numpy as np


#Define model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=4)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=4)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20*10*10, 50)
        self.fc2 = nn.Linear(50, 40)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()
optimizer= optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()


def train(epoch, train_loader):
    model.train()
    train_loss = 0
    avr_loss = []
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = Variable(image), Variable(label)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        avr_loss.append(loss.data)
        train_loss += criterion(output, label).data
        print(loss.data)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(image), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data))
    train_loss /= len(train_loader.dataset)
    print(avr_loss)
    print('Loss: {:.6f}'.format(np.mean(avr_loss)))

    result = train_loss

    return result

def test(test_loader):
    correct = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for (image, label) in test_loader:
            image, label = Variable(image.float()), Variable(label)

            output = model(image)
            test_loss += criterion(output, label).data
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss = test_loss
    accuracy = 100. * correct / len(test_loader.dataset)

    return loss, accuracy

def evaluate(test_loader, mod):
    model = Net()
    # load saved parameters
    param = torch.load(mod)
    model.load_state_dict(param)
    # evaluation mode
    model.eval()

    labels = []
    probs = []
    pred = []
    Y = []
    for i, (x,y) in enumerate(test_loader):
        with torch.no_grad():
            output = model(x)
            prob, label = torch.topk(output, 2)

        labels.append(label.tolist())
        probs.append(torch.exp(prob).tolist())


        pred += [int(l.argmax()) for l in output]
        Y += [int(l) for l in y]

    print(classification_report(Y, pred)) # https://scrapbox.io/nishio/classification_reportの読み方
    print(pred)
    print(len(Y))
    print(labels)
    print(probs)

# train
def learn(train_loader, test_loader, epoch):
    train_result = []
    train_results = []
    for ep in range(int(epoch)):
        train_results.append(train(ep, train_loader))

    torch.save(model.state_dict(), 'cnn_dict.model')
    torch.save(model, 'cnn.model')

    train_result = (np.mean(train_results))
    test_result = test(test_loader)


    return train_result, test_result, epoch
