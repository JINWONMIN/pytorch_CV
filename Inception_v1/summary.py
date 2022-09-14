import config
from torchsummary import summary
from GoogLeNet import *


def model_summary():
    model = GoogLeNet()

    model.to(config.device)

    return summary(model, (3, 96, 96))


if __name__ == "__main__":
    model_summary()
