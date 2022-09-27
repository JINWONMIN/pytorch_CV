import torch.nn as nn
# import torch.nn.functional as F

from utils import Params


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes):
        super(VGG, self).__init__()

        self.params = Params('/content/drive/MyDrive/VGG-16/config/config.yml')
        self.cfg = self.params.cfg

        self.vgg_name = vgg_name
        self.features = self._make_layers(self.cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(True)    # Reduce the memory usage
                           ]
                in_channels = x
        return nn.Sequential(*layers)


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, num_classes):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, inputs):
        x1 = self.modelA(inputs)
        x2 = self.modelB(inputs)
        x = x1 + x2
        x = nn.Softmax(dim=1)(x)
        return x
