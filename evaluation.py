from data_gen import CellTestDataset
from model import get_instance_segmentation_model
import os
import json
import transforms as T
import torch
from PIL import Image
import torchvision.transforms as tfs
import pycocotools._mask as _mask
import numpy as np
import copy


def encode(bimask):
    if len(bimask.shape) == 3:
        return _mask.encode(bimask)
    elif len(bimask.shape) == 2:
        h, w = bimask.shape
        return _mask.encode(bimask.reshape((h, w, 1), order='F'))[0]


def decode(rleObjs):
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:, :, 0]


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)

if __name__ == "__main__":
    dataset_test = \
    CellTestDataset('dataset/test', get_transform(train=False))
    result_to_json = []
    model = get_instance_segmentation_model(num_classes=2)
    model.load_state_dict(torch.load("model.pkl"), strict=False)
    model.eval()
    device = torch.device('cuda:7') \
        if torch.cuda.is_available() else torch.device('cpu')
    model = torch.nn.DataParallel(model, device_ids=[7, 6, 8, 9])
    model.to(device)

    for id in range(6):
        print(id)

        img = dataset_test[id]
        with torch.no_grad():
            prediction = model([img.to(device)])

        prediction = prediction[0]

        image_id = id+1
        category_id = 1
        boxes = prediction["boxes"].cpu()
        masks = prediction["masks"].cpu()
        scores = prediction["scores"].cpu()
        masks = np.array(masks, dtype=np.float32)
        masks = np.where(masks > 0.5, True, masks)
        masks = np.where(masks <= 0.5, False, masks)
        masks = np.array(masks, dtype=np.uint8)
        num, _, h, w = masks.shape
        masks = masks.reshape(num, h, w, 1)

        for i in range(boxes.shape[0]):
            det_box_info = {}
            det_box_info["image_id"] = image_id
            bbox = boxes[i]
            det_box_info["bbox"] = [
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]-bbox[0]),
                float(bbox[3]-bbox[1])
                ]
            det_box_info["score"] = float(scores[i])
            det_box_info["category_id"] = 1

            det_box_info["segmentation"] = \
            encode(np.asfortranarray(masks[i]))[0]
            det_box_info["segmentation"]["counts"] = \
            det_box_info["segmentation"]["counts"].decode('ascii')
            result_to_json.append(det_box_info)

        json_object = json.dumps(result_to_json, indent=4)

        with open("answer.json", "w") as outfile:
            outfile.write(json_object)
