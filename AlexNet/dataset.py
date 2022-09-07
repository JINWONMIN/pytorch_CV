import sys
from glob import glob

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import find_classes
from torchvision.transforms import TrivialAugmentWide

import imgproc


# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")

# The delimiter is not the same between different platforms
if sys.platform == "win32":
    delimiter = "\\"
else:
    delimiter = "/"


class ImageDataset(Dataset):
    """Define training/valid dataset loading methods.
    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): Image size.
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir: str, image_size: int, mode: str) -> None:
        super(ImageDataset, self).__init__()
        # Iterate over all image paths
        self.image_file_paths = glob(f"{image_dir}/*/*")
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(image_dir)
        self.image_size = image_size
        self.mode = mode
        self.delimiter = delimiter

        if self.mode == "Train":
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                TrivialAugmentWide(),
                transforms.RandomRotation([0, 270]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ])
        elif self.mode == "Valid" or self.mode == "Test":
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop([self.image_size, self.image_size]),
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"

        self.post_transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]:
        image_dir, image_name = self.image_file_paths[batch_index].split(self.delimiter)[-2:]
        # Read a batch of image data
        if image_name.split(".")[-1].lower() in IMG_EXTENSIONS:
            image = cv2.imread(self.image_file_paths[batch_index])
            target = self.class_to_idx[image_dir]
        else:
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, "
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # OpenCV convert PIL
        image = Image.fromarray(image)

        # Data preprocess
        image = self.pre_transform(image)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = imgproc.image_to_tensor(image, False, False)

        # Data postprocess
        tensor = self.post_transform(tensor)

        return {"image": tensor, "target": target}

    def __len__(self) -> int:
        return len(self.image_file_paths)
