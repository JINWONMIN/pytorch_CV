import os

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.utils.tensorboard import SummaryWriter

from model import AlexNet, AlexNetMulti
from utils import make_directory, define_scheduler, define_loss, define_optimizer
from dataset import ImageDataset

from tqdm.notebook import tqdm
import time

import config


### DataLoader
# Load train and valid datasets
train_set = ImageDataset(config.train_image_dir, config.image_size, "Train")
valid_dataset = ImageDataset(config.valid_image_dir, config.image_size, "Valid")

# Generator all dataloader
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)


### weights init
def init_weights(m):
    if type(m) not in [nn.ReLU, nn.LocalResponseNorm, nn.MaxPool2d, nn.Sequential, nn.Dropout, AlexNet]:
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        # original paper  = 1 for Conv2d layers
        m.bias.data.fill_(1)

### Create log dir
make_directory()

### Create training process log file
writer = SummaryWriter(os.path.join())


### consist of train and test
def train():
    if config.model_arch_name == "alexnet":
        alexnet = AlexNet(num_classes=config.model_num_classes)
        alexnet.apply(init_weights)
        alexnet = alexnet.to(config.device)
        print("AlexNet created")

    elif config.model_arch_name == "AlexNetMulti":
        alexnet = AlexNetMulti(num_classes=config.model_num_classes)
        alexnet.apply(init_weights)
        alexnet = alexnet.to(config.device)
        print("AlexNetMulti created")

    criterion = define_loss()
    optimizer = define_optimizer(alexnet)
    print('Optimizer created')
    lr_scheduler = define_scheduler(optimizer)
    print('LR Scheduler created')

    start_time = time.time()
    min_loss = int(1e9)
    history = []
    count = 0

    print("Starting training...")
    total_steps = 1
    for epoch in range(config.epochs):
        lr_scheduler.step()
        epoch_loss = float(0)
        tk0 = tqdm(train_dataloader, total=len(train_dataloader), leave=False)
        for step, (inputs, labels) in enumerate(tk0, 0):
            inputs, labels = inputs.to(config.device), labels.to(config.device)

            # calculate the loss
            outputs = alexnet(inputs)
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log the information and add to tensorboard
            epoch_loss += loss.item()
            history.append(loss.item())

        # validataion
        class_correct = list(0. for i in range(1000))
        class_total = list(0. for i in range(1000))
        with torch.no_grad():
            for data in valid_dataloader:
                images, labels = data
                images = images.to(config.device)
                labels = labels.to(config.device)
                outputs = alexnet(images)
                _, pred = torch.max(outputs, 1)
                c = (pred == labels).squeeze()
                for i in range(labels.size()[0]):
                    label = labels[i].item()
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        tqdm.write('[Epoch : %d] train_loss: %.5f val_acc: %.2f' %
                   (epoch + 1, epoch_loss / 157, sum(class_correct) / sum(class_total) * 100))
        if min_loss > epoch_loss:
            count += 1
            if count > 10:
                for g in optimizer.param_groups:
                    g['lr'] /= 10
        else:
            min_loss = epoch_loss
            count = 0

    print(time.time()-start_time)
    print('Finished Training')

