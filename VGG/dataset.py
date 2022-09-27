import os, glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision
import torchvision.transforms as transforms

from utils import Params


class SDataset(Dataset):
    def __init__(self):
        self.params = Params('/content/drive/MyDrive/VGG-16/config/config.yml')

        self.path = self.params.path

    def load_data(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = torchvision.datasets.STL10(root=self.path, split='train', download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=self.params.batch_size, shuffle=True)

        testset = torchvision.datasets.STL10(root=self.path, split='test', download=True, transform=transform)
        test_loader = DataLoader(testset, batch_size=self.params.batch_size, shuffle=False)

        return train_loader, test_loader


class CdDataset(Dataset):
    def __init__(self, transform=None, file_paths=None):
        self.params = Params('/content/drive/MyDrive/VGG-16/config/config.yml')

        self.path = self.params.path
        self.folder = os.listdir(self.path)

        self.train_path = []
        self.val_path = []
        self.test_path = []
        self.li = [self.train_path, self.val_path, self.test_path]

        self.transform = transform
        self.file_paths = file_paths
        self.Image_List = []
        self.Label_List = []

    def prepare_data(self):
        for idx, folder in enumerate(self.folder):
            folder_path = os.path.join(self.path, folder)
            folder_path_list = os.listdir(folder_path)
            for fp in folder_path_list:
                folder_path2 = os.path.join(folder_path, fp)
                files = glob.glob(folder_path2 + '/*.jpg')
                self.li[idx] += files
        return self.li[0], self.li[1], self[2]

    def load_data(self):
        # labeling the dataset
        for i in range(len(self.file_paths)):
            if 'dog' in self.file_paths[i]:
                self.Image_List.append(self.file_paths[i])
                self.Label_List.append(1)
            elif 'cat' in self.file_paths[i]:
                self.Image_List.append(self.file_paths[i])
                self.Label_List.append(0)

    def __len__(self):
        return len(self.Label_List)

    def __getitem__(self, idx):
        label = self.Label_List[idx]
        img = Image.open(self.Image_List[idx].convert('RGB'))
        if self.transform:
            img = self.transform(img)
        return img, label
