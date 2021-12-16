import utils
import transforms as T
import torch
import torchvision.transforms as tfs
from engine import train_one_epoch, evaluate, validation
from data_gen import CellTrainDataset
from model import get_instance_segmentation_model
import numpy as np
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)

# use the PennFudan dataset and defined transformations
dataset_train = CellTrainDataset('../cell_data', get_transform(train=True))
dataset_valid = CellTrainDataset('../cell_data', get_transform(train=False))

torch.manual_seed(1)
indices = torch.randperm(len(dataset_train)).tolist()
dataset_train = torch.utils.data.Subset(dataset_train, indices[:-2])
dataset_valid = torch.utils.data.Subset(dataset_valid, indices[-2:])


# define training and validation data loaders
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
data_loader_valid = torch.utils.data.DataLoader(
    dataset_valid, batch_size=2, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

device = torch.device('cuda:8') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2
model = get_instance_segmentation_model(num_classes)
model.to(device)
model = torch.nn.DataParallel(model, device_ids=[8, 7, 6, 9])
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=1e-3,
                            momentum=0.9, weight_decay=0.0005)
min = 2.5107
# the learning rate scheduler decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


num_epochs = 100
model.load_state_dict(torch.load("model.pkl"), strict=False)
for epoch in range(num_epochs):
    
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)

    # update the learning rate
    lr_scheduler.step()
    loss = validation(model, data_loader_valid, device, epoch, print_freq = 10)
    print("valid loss:", loss)
    
    if loss < min:
        min = loss
        torch.save(model.state_dict(), 'model.pkl')
        print("save model")
