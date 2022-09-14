from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets

import config


def train_cifar_dataloader():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    train_dataset = datasets.CIFAR10(config.train_path, train=True, download=True, transform=transform)

    # Split dataset into training set and validation set
    train_dataset, val_dataset = random_split(train_dataset, (config.train_samples_num, config.val_samples_num))

    print("image shape of a random sample image : {}".format(train_dataset[0][0].numpy().shape), end='\n\n')

    print(f"Training set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")

    # Generate dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    return train_loader, val_loader


def test_cifar_dataloader():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    test_dataset = datasets.CIFAR10(config.test_path, train=False, download=True, transform=transform)

    print(f"Training set: {len(test_dataset)} images")

    # Generate dataloader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    return test_loader
