import torch
import torch.nn as nn
import torch.optim as optim

import time

from dataset import SDataset
from model import VGG, MyEnsemble
from utils import Params


def train():
    params = Params('/content/drive/MyDrive/VGG-16/config/config.yml')
    Dataset = SDataset()

    classes = params.classes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = Dataset.load_data()

    modelA = VGG(vgg_name=params.model_name, num_classes=params.num_classes).to(device)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(modelA.parameters(), lr=params.lr)

    start_time = time.time()
    for epoch in params.epochs:
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = modelA(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    print(time.time() - start_time)
    print('Finished Training')

    class_correct = list(0. for i in params.epochs)
    class_total = list(0. for i in params.epochs)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = modelA(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(params.num_classes):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def ensemble():
    # Create models and load state_dicts
    params = Params('/content/drive/MyDrive/VGG-16/config/config.yml')
    Dataset = SDataset()

    classes = params.classes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = Dataset.load_data()

    modelA = VGG(vgg_name=params.model_name, num_classes=params.num_classes).to(device)
    modelB = VGG(vgg_name=params.model_name, num_classes=params.num_classes).to(device)
    # Load state dicts
    modelA.load_state_dict(torch.load('/content/drive/MyDrive/Study_AI/vgg11.pth'))
    modelB.load_state_dict(torch.load('/content/drive/MyDrive/Study_AI/vgg11_B.pth'))

    model = MyEnsemble(modelA, modelB, num_classes=params.num_classes)
    model = model.to(device)

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == "__main__":
    train()
