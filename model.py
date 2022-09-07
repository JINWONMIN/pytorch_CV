import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, init_weights: bool = True):
        super(AlexNet, self).__init__()
        self.convnet = nn.Sequential(
            # Input Channel (RGB: 3)
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, padding=0, stride=4),     # 227 -> 55
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),    # 55 -> 27

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, stride=1),    # 27 -> 27
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),      # 27 -> 13

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),      # 13 -> 6
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.fclayer = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor):
        out = self.convnet(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fclayer(out)

        return out


# Multi GPU가 아니기 때문에 논리 구조상 맞도록 Input, Output을 조절
class AlexNetMulti(nn.Module):
    def __init__(self, num_classes: int = 1000, init_weights: bool = True):
        super(AlexNetMulti, self).__init__()
        self.fstblock_1 = nn.Sequential(
            # Input Channel (RGB: 3)
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, padding=0, stride=4),  # 227 -> 55
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 55 -> 27
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, padding=2, stride=1),  # 27 -> 27
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27 -> 13
        )
        self.fstblock_2 = nn.Sequential(
            # Input Channel (RGB: 3)
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, padding=0, stride=4),  # 227 -> 55
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 55 -> 27
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, padding=2, stride=1),  # 27 -> 27
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27 -> 13
        )

        self.cross_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
        )
        self.cross_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
        )

        self.sndblock_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13 -> 6
        )
        self.sndblock_2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13 -> 6
        )

        self.crossfc1_1 = nn.Sequential(
            nn.Linear(128 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.crossfc1_2 = nn.Sequential(
            nn.Linear(128 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        self.crossfc2_1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.crossfc2_2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        self.classifier_1 = nn.Linear(2048, num_classes)
        self.classifier_2 = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor):
        # First Block
        x1 = self.fstblock_1(x)
        x2 = self.fstblock_2(x)

        # Cross
        x3 = self.cross_conv_1(x1)  # Left Block 1
        x4 = self.cross_conv_2(x2)  # Left Block 2

        x5 = self.cross_conv_1(x1)  # Right Block 1
        x6 = self.cross_conv_2(x2)  # Right Block 2

        x1 = torch.cat([x3, x4], 1)
        x2 = torch.cat([x5, x6], 1)

        # Second Block
        x1 = self.sndblock_1(x1)
        x2 = self.sndblock_2(x2)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        # FC Layer (Cross)
        x3 = self.crossfc1_1(x1)  # Left FC 1
        x4 = self.crossfc1_2(x2)  # Left FC 2

        x5 = self.crossfc1_1(x1)  # Right FC 1
        x6 = self.crossfc1_2(x2)  # Right FC 2

        x1 = torch.cat([x3, x4], 1)
        x2 = torch.cat([x5, x6], 1)

        # FC Layer (Cross)
        x3 = self.crossfc2_1(x1)  # Left FC 1
        x4 = self.crossfc2_2(x2)  # Left FC 2

        x5 = self.crossfc2_1(x1)  # Right FC 1
        x6 = self.crossfc2_2(x2)  # Right FC 2

        x1 = torch.cat([x3, x4], 1)
        x2 = torch.cat([x5, x6], 1)

        x1 = self.classifier_1(x1)
        x2 = self.classifier_2(x2)

        x = (x1 + x2) / 2
        return x
