import os
import random
import shutil

from utils import Params


def split_data(file_path, val_path, test_path):
    files = os.listdir(file_path)

    dir_list = [val_path, test_path]

    for path in dir_list:
        target_list = random.sample(files, k=4000)
        for img in target_list:
            shutil.move(file_path + '/' + img, path + '/' + img)


if __name__ == "__main__":
    params = Params('./config/config.yml')

    file_path = params.train_path
    val_path = params.val_path
    test_path = params.val_path

    split_data(file_path, val_path, test_path)
