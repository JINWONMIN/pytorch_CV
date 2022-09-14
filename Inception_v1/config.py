import torch
import torch.nn as nn


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

## Path
train_path = '/content/drive/MyDrive/inception_v1/data/train'
test_path = '/content/drive/MyDrive/inception_v1/data/test'
save_path = '/content/drive/MyDrive/inception_v1/sample_data/'

## Dataset
train_samples_num = 45000
val_samples_num = 5000
test_samples_num = 10000

## hyperparameter
epoch = 15
batch_size = 128
lr = 1e-3

criterion = nn.CrossEntropyLoss()