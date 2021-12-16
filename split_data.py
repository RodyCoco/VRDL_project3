import os
import torch
import numpy as np
import torch.utils.data
import shutil
from PIL import Image

if not os.path.exists("dataset/train_images"):
    os.mkdir("dataset/train_images")

train_img_path = "dataset/train"
train_img_dir = os.listdir(train_img_path)
for img_name in train_img_dir:
    if img_name[-3] != "png":
        continue
    shutil.copyfile(
        os.path.join(train_img_path, img_name, "images", img_name+".png"),
        os.path.join("dataset/train_images", img_name+".png"),
        )

