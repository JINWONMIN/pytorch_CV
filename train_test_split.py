import os
import shutil
import numpy as np
from glob import glob
import shutil


class Split():
    def __init__(self, train_dest: str, test_dest: str, val_dest: str, source_train: str,
                 source_test: str, source_val: str):
        self.train_dest = train_dest
        self.test_dest = test_dest
        self.val_dest = val_dest
        self.source_train = source_train
        self.source_test = source_test
        self.source_val = source_val

    def make_directory(self, dir_path: str) -> None:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def move_image(self, file_list: list, dir_dest: str) -> None:
        for f in file_list:
            file_name = os.path.basename(f)
            shutil.copy(f, dir_dest + '/' + file_name)

    def split_image(self, file_list: list, dir_dest: str, ratio: float) -> None:
        for f in file_list:
            file_name = os.path.basename(f)
            if np.random.rand(1) < ratio:
                shutil.move(f, dir_dest + '/' + file_name)

    def main(self):
        self.make_directory(self.train_dest)
        self.make_directory(self.val_dest)
        self.make_directory(self.test_dest)

        train_list = glob(self.source_train, recursive=True)
        val_list = glob(self.source_val, recursive=True)

        self.move_image(train_list, self.train_dest)
        self.move_image(val_list, self.val_dest)

        test_list = glob(self.source_val, recursive=True)
        self.split_image(test_list, self.test_dest, ratio=0.2)


if __name__ == "__main__":
    train_dest = "/content/drive/MyDrive/AlexNet/data/train"
    val_dest = "/content/drive/MyDrive/AlexNet/data/val"
    test_dest = "/content/drive/MyDrive/AlexNet/data/test"

    source_train = "/content/drive/MyDrive/AlexNet/data/train_ori/**/*.JPEG"
    source_val = "/content/drive/MyDrive/AlexNet/data/val_ori/**/*.JPEG"
    source_test = "/content/drive/MyDrive/AlexNet/data/val/*.JPEG"

    Split(train_dest=train_dest, val_dest=val_dest, test_dest=test_dest,
          source_train=source_train, source_val=source_val, source_test=source_test)
