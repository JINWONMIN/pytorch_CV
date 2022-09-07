import os

from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from torchvision import transforms, datasets

import config


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def define_loss() -> nn.CrossEntropyLoss:
    criterion = nn.CrossEntropyLoss(label_smoothing=config.loss_label_smoothing)
    criterion = criterion.to(device=config.device)

    return criterion


def define_optimizer(model):
    optimizer = optim.SGD(model.parameters(),
                          lr=config.model_lr,
                          momentum=config.model_momentum,
                          weight_decay=config.model_weight_decay)

    return optimizer


def define_scheduler(optimizer: optim.SGD) -> lr_scheduler.CosineAnnealingWarmRestarts:
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                         config.lr_scheduler_T_0,
                                                         config.lr_scheduler_T_mult,
                                                         config.lr_scheduler_eta_min)

    return scheduler
