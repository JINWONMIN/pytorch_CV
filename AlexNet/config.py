import random
import numpy as np

import torch
from torch.backends import cudnn



random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Turning on when the image size does not chane during training can speed up training
cudnn.benchmark = True
# Model number class
model_num_classes = 1000
# Current configuration parameter method
mode = "train"
# Model architecture name
model_arch_name = "alexnet"     # ["alexnet", "alexnetmulti"]
# Experiment name, easy to save weights and log files
exp_name = f"{model_arch_name.upper()}-ImageNet_1K"
# log dir path


if mode == "train":
    # output dir
    output_dir = "/content/drive/MyDrive/AlexNet/results"

    # Dataset address
    train_image_dir = "/content/drive/MyDrive/AlexNet/data/train"
    valid_image_dir = "/content/drive/MyDrive/AlexNet/data/val"

    image_size = 224
    batch_size = 128
    num_workers = 4

    epochs = 600

    # Loss parameters
    loss_label_smoothing = 0.1

    # Optimizer parameter
    model_lr = 0.5
    model_momentum = 0.9
    model_weight_decay = 2e-05
    model_ema_decay = 0.99998

    # Learning rate scheduler parameter
    lr_scheduler_T_0 = epochs // 4
    lr_scheduler_T_mult = 1
    lr_scheduler_eta_min = 5e-5

    # How many iterations to print the training/validate result
    train_print_frequency = 200
    valid_print_frequency = 20


if mode == "test":
    # Test data address
    test_image_dir = "/content/drive/MyDrive/AlexNet/data/test"

    # Test dataloader parameters
    image_size = 224
    batch_size = 256
    num_workers = 4

    # How many iterations to print the testing result
    test_print_frequency = 20

    model_weights_path = ""
