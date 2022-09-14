import torch

import config
from GoogLeNet import *
from dataset import test_cifar_dataloader


def evaluate():
    test_loader = test_cifar_dataloader()

    model = GoogLeNet()
    model.load_state_dict(torch.load(config.save_path + 'googlenet_model'))

    num_test_samples = config.test_samples_num
    correct = 0

    model.eval().cuda()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)

            # Make predictions
            prediction, _, _ = model(inputs)

            # Retrieve predictions indexes.
            _, predicted_class = torch.max(prediction.data, 1)

            # Compute number of correct predictions
            correct += (predicted_class == labels).float().sum().item()

    test_accuracy = correct / num_test_samples

    print(f'Test accuracy: {test_accuracy}')


if __name__ == "__main__":
    evaluate()
