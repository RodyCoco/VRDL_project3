import os
import torch
import numpy as np
import torch.utils.data
import shutil
from PIL import Image

root = "../cell_data/train"
root_dir = os.listdir(root)
print(root_dir)

file = open("../cell_data/train_img_id.txt", "a")
for s in root_dir:
    if "TCGA" in s:
        file.write(s+"\n")

if not os.path.exists("../cell_data/train/images"):
    os.mkdir("../cell_data/train/images")
if not os.path.exists("../cell_data/train/masks"):
    os.mkdir("../cell_data/train/masks")


file = open("../cell_data/train_img_id.txt", "r")

for img_name in file.readlines():
    img_name = img_name[:-1]
    img_path = os.path.join(root, img_name, "images",f"{img_name}.png")
    shutil.copyfile(img_path, os.path.join(root, "images",f"{img_name}.png"))
# for img_name in os.listdir("cell_data/train"):
#     mask_dir = os.path.join(root, img_name, "masks")
#     shutil.copytree("data/train/masks", mask_dir)