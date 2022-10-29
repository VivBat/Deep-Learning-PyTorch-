import numpy as np
import torch
from torch import nn

import requests
import zipfile
from pathlib import Path
import os

import random
from PIL import Image

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# download the folder containing images if it doesn't exist
if image_path.is_dir():
    print(f"Directory {image_path} already exists")
else:
    image_path.mkdir(parents=True, exist_ok=True)

    # download the data from a URL
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading data....")
        f.write(request.content)

    # Unzipping the folder containing data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping data...")
        zip_ref.extractall(image_path)


def walk_through_dir(dir_path):
    """
    Walks through all the directories and files of a given path
    :param dir_path: target directory
    :return: Prints out:
                number of subdirectories in dir_path
                number of images in each subdirectory
                name of each subdirectory
    """
    # print(list(os.walk(dir_path)))
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(filenames)} images and {len(dirnames)} directories in {dirpath}")


walk_through_dir(image_path)

train_dir = image_path / "train"
test_dir = image_path / "test"

# Visualising an image randomly picked from the dataset
# random.seed(42)

# getting all the possible paths of the images
image_path_list = list(image_path.glob("*/*/*.jpg"))

# print(len(image_path_list))

# choosing an image path randomly
random_image_path = random.choice(image_path_list)
print(f"Random image path: {random_image_path}")

# image class
image_class = random_image_path.parent.stem
print(f"Image class: {image_class}")

# open image
img = Image.open(random_image_path)

print(f"Image height: {img.height}")
print(f"IMage width: {img.width}")

img.show()      # show the image

# # plotting using matplotlib
# image_as_array = np.asarray(img)
# plt.figure()
# plt.imshow(image_as_array)
# plt.title(f"Image class: {image_class} | Image shape: {image_as_array.shape}")
# plt.axis(False)
# plt.show()



