import os
import torch
import numpy as np
import torch.utils.data
from PIL import Image
import torchvision.transforms as tfs
import matplotlib.pyplot as plt


class CellTrainDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs =
        list(sorted(os.listdir(os.path.join(root, "train_images"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "train_images", self.imgs[idx])
        mask_dir_name = self.imgs[idx][:-4]
        mask_path = os.path.join(self.root, "train", mask_dir_name, "masks")
        img = Image.open(img_path).convert("RGB")

        num = len(os.listdir(mask_path))
        w, h = img.size
        masks = np.zeros((num, h, w))  # size: (mask_num,1 ,h, w)
        mask_dir = os.listdir(mask_path)
        for idx, mask_img in enumerate(mask_dir):
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            mask = Image.open(os.path.join(mask_path, mask_img))
            mask = np.array(mask)
            mask = np.where(mask == 255, True, mask)
            mask = np.where(mask == 0, False, mask)
            masks[idx] = mask

        num_objs = num
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class CellTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = [
                    "TCGA-A7-A13E-01Z-00-DX1.png",
                    "TCGA-50-5931-01Z-00-DX1.png",
                    "TCGA-G2-A2EK-01A-02-TSB.png",
                    "TCGA-AY-A8YK-01A-01-TS1.png",
                    "TCGA-G9-6336-01Z-00-DX1.png",
                    "TCGA-G9-6348-01Z-00-DX1.png"
                    ]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        T = tfs.ToTensor()
        if self.transforms is not None:
            img = T(img)

        return img

    def __len__(self):
        return len(self.imgs)
