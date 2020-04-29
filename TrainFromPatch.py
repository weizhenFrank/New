from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from skimage import io, transform
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet
import PIL
import torch.multiprocessing
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

model_name = 'efficientnet-b1'
torch.multiprocessing.set_sharing_strategy('file_system')
# writer_path = '/p300/Tboard_try_paper'
writer_path = '/p300/TensorBoardTumorType/Output/3lr0.0001decay0.001Down1.0BalanceDropOut0.2'
# data_dir = '/p300/hymenoptera_data'
# data_dir = '/p300/liuwzh_p300/AntsAndBees'
data_dir = '/p300/TumorType20XImage128/'
NumClass = 2
BatchSize = 128
NumWorkers = 16
DownsamplePCT = 1.0
learning_rate = 0.0001
dropout_rate = 0.2
WeightDecay = 0.001
StepSize = 10
Gamma = 0.1
image_size = EfficientNet.get_image_size(model_name)
writer = SummaryWriter(writer_path)
Momentum = 0.9
Epoch = 20


def get_subset(indices, start, end):
    return indices[start: start + end]


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

train_count = int(len(image_datasets['train']) * DownsamplePCT)
train_indices = torch.randperm(len(image_datasets['train']))
train_index = get_subset(train_indices, 0, train_count)

val_count = int(len(image_datasets['val']) * DownsamplePCT)
val_indices = torch.randperm(len(image_datasets['val']))
val_index = get_subset(val_indices, 0, val_count)

dataloaders = {
    "train": torch.utils.data.DataLoader(
        image_datasets['train'], sampler=torch.utils.data.SubsetRandomSampler(train_index), batch_size=BatchSize,
        num_workers=NumWorkers
    ),
    "val": torch.utils.data.DataLoader(
        image_datasets['val'], sampler=torch.utils.data.SubsetRandomSampler(val_index), batch_size=BatchSize,
        num_workers=NumWorkers
    ),
}

dataset_sizes = {"train": train_count, "val": val_count}
class_names = image_datasets['train'].classes


def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        roc_auc_value = {}
        acc_value = {}
        loss_value = {}
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            class_probs = []
            class_preds = []
            class_labels = []
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    class_probs_batch = [F.softmax(el, dim=0) for el in outputs]
                    class_probs.append(class_probs_batch)
                    class_preds.append(preds)
                    class_labels.append(labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
            test_preds = torch.cat(class_preds)
            test_labels = torch.cat(class_labels)
            test_labels_np = test_labels.detach().cpu().numpy()
            test_probs_np = test_probs.detach().cpu().numpy()
            fpr, tpr, _ = roc_curve(test_labels_np, test_probs_np[:, 1])

            roc_auc = auc(fpr, tpr)
            roc_auc_value[phase] = roc_auc
            acc_value[phase] = epoch_acc
            loss_value[phase] = epoch_loss
            # writer.add_scalar('Loss/'+phase, epoch_loss, epoch)
            # writer.add_scalar('Accuracy'+phase, epoch_acc, epoch)

            # writer.add_scalars('Loss', {'Train': epoch_loss if phase == 'train',
            #                             'Val': epoch_loss if phase == 'val'}, epoch)
            # writer.add_scalars('Loss', {'Train': epoch_loss if phase == 'train',
            #                             'Val': epoch_loss if phase == 'val'}, epoch)
            #
            # writer.flush()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        # writer.add_scalar('Loss/train', loss_value['train'], epoch)
        # writer.add_scalar('Loss/val', loss_value['val'], epoch)
        # writer.add_scalar('Acc/train', acc_value['train'], epoch)
        # writer.add_scalar('Acc/val', acc_value['val'], epoch)
        writer.add_scalars('run3/AUC', {'Train': roc_auc_value['train'],
                                           'Val': roc_auc_value['val']}, epoch)
        writer.add_scalars('run3/Loss', {'Train': loss_value['train'],
                                            'Val': loss_value['val']}, epoch)
        writer.add_scalars('run3/Acc', {'Train': acc_value['train'],
                                           'Val': acc_value['val']}, epoch)
        #
        writer.flush()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = EfficientNet.from_pretrained(model_name, num_classes=NumClass)
num_ftrs = model_ft._fc.in_features
model_ft._fc = nn.Linear(num_ftrs, NumClass)
model_ft._dropout = nn.Dropout(dropout_rate)
# print(model_ft._dropout)
### Freeze the 1/2 layers
# for name,param in model_ft.named_parameters():
#     if name.split(".")[1] == "12":
#         break
#     else:
#         param.requires_grad = False


# model_ft_para=[]
# for name,param in model_ft.named_parameters():
#     model_ft_para.append(name)
#
# del model_ft_para[-2:]

### Freeze the layers except last four layer
# for name,param in model_ft.named_parameters():
#     if name in model_ft_para:
#         param.requires_grad = False


params_to_update = model_ft.parameters()
print("Params to learn:")

if True:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# model_ft = EfficientNet.from_pretrained(model_name,num_classes=NumClass)
# model_ft._fc = nn.Linear(1000, 2)
# model_ft = model_ft.to(device)
#gpu = "4,5,6,7"
gpu = "0,1,2,3"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
model_ft = nn.DataParallel(model_ft).cuda()

# os.environ['CUDA_VISIBLE_DEVICES'] = gpu
# model_ft = nn.Dataparalle(model_ft).cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=Momentum, weight_decay=WeightDecay)
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=WeightDecay, amsgrad=False)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=StepSize, gamma=Gamma)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=Epoch)

# torch.save(model_ft, "/p300/TrainedBySmallDataModel.pth")
