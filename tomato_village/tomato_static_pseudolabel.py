# -*- coding: utf-8 -*-
"""
At this point we're throwing all the data we can in.  This has a bunch of extra
tomato images in addition to the labeled and (statically) pseudo-labeled
images we were provided with.
"""

import os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from PIL import Image
import torch
import tqdm
import platform
import shutil
import json
import timm
from timm.loss import LabelSmoothingCrossEntropy
import matplotlib.pyplot as plt
import yaml
from augment import new_data_aug_generator

# https://github.com/facebookresearch/deit/blob/7e160fe43f0252d17191b71cbb5826254114ea5b/datasets.py#L108
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Michelle
#data_prefix = "/media/nvme1/mitquach/ucsc-cse-244-a-2024-fall-final-project/"
#model_prefix = "/media/nvme1/mitquach/ucsc-cse-244-a-2024-fall-final-project/models/"
data_prefix = "/soe/jcasper/244A_Final_Project"
#model_prefix = "/soe/jcasper/244A_Final_Project/models"
model_prefix = "/projects/inrg-lab/244A_models"

tomato_prefix = "/projects/inrg-lab/tomato/Variant-a(Multiclass Classification)/"

#if platform.node() == 'navi': # Daniel
#    data_prefix = "/home/argon/Stuff/CSE244A_project/"
#    model_prefix = "/home/argon/Stuff/CSE244A_project/models/"

categories = pd.read_csv(os.path.join(data_prefix, 'tomato_ext_categories.csv'))
train_labels = pd.read_csv(os.path.join(data_prefix, 'train_labeled.csv'))

tomato_labels = pd.read_csv(os.path.join(data_prefix, 'tomato.csv'))

def save_config(conf):
    with open(conf["model_name"] + ".yaml", 'w') as f:
        yaml.dump(conf, f)

def load_config(conf_name):
    with open(conf_name + ".yaml", 'r') as f:
        return yaml.safe_load(f)

# Someday this should probably be some yaml files... - Daniel

#TODO: Add batch size to this
#TODO: Add model type (e.g. facebookresearch/deit:main)

# training_config = {
#     "model_name":  "michelle_diet__freeze11__plateaulr_0.1_0_0.0__AdamW_wdecay_1en4",
#     "optimizer_type": "AdamW",
#     "optimizer_lr": 0.0001,
#     "optimizer_wd": 0.0001,
#     "scheduler_type": "ReduceLROnPlateau",
#     "scheduler_params": {"factor": 0.1,
#                          "patience": 0,
#                          "threshold": 0.0},
#     "unfreeze_layers": ['blocks.11.norm1.weight', 'blocks.11.norm1.bias', 'blocks.11.attn.qkv.weight', 'blocks.11.attn.qkv.bias', 'blocks.11.attn.proj.weight', 'blocks.11.attn.proj.bias', 'blocks.11.norm2.weight', 'blocks.11.norm2.bias', 'blocks.11.mlp.fc1.weight', 'blocks.11.mlp.fc1.bias', 'blocks.11.mlp.fc2.weight', 'blocks.11.mlp.fc2.bias', 'norm.weight', 'norm.bias', 'head.weight', 'head.bias']
# }
# save_config(training_config)
# training_config = load_config("michelle_diet__freeze11__plateaulr_0.1_0_0.0__AdamW_wdecay_1en4")

# training_config = load_config("michelle_diet_imagenetmean__freeze11__plateaulr_0.1_0_0.0__AdamW_wdecay_1en4")
# training_config = load_config("michelle_diet_imagenetmean_augmentD__freeze11__explr_1en4_0.8__AdamW_wdecay_1en4")

# training_config = {
#     "model_name":  "michelle_diet_imagenetmean_augmentD__freeze10__explr_1en4_0.8__AdamW_wdecay_1en4",
#     "optimizer_type": "AdamW",
#     "optimizer_lr": 0.0001,
#     "optimizer_wd": 0.0001,
#     "augment_mode": "augmentD",
#     "scheduler_type": "ExponentialLR",
#     "scheduler_params": {"gamma": 0.8,},
#     "unfreeze_layers": ['blocks.10.norm1.weight', 'blocks.10.norm1.bias', 'blocks.10.attn.qkv.weight', 'blocks.10.attn.qkv.bias', 'blocks.10.attn.proj.weight', 'blocks.10.attn.proj.bias', 'blocks.10.norm2.weight', 'blocks.10.norm2.bias', 'blocks.10.mlp.fc1.weight', 'blocks.10.mlp.fc1.bias', 'blocks.10.mlp.fc2.weight', 'blocks.10.mlp.fc2.bias', 'blocks.11.norm1.weight', 'blocks.11.norm1.bias', 'blocks.11.attn.qkv.weight', 'blocks.11.attn.qkv.bias', 'blocks.11.attn.proj.weight', 'blocks.11.attn.proj.bias', 'blocks.11.norm2.weight', 'blocks.11.norm2.bias', 'blocks.11.mlp.fc1.weight', 'blocks.11.mlp.fc1.bias', 'blocks.11.mlp.fc2.weight', 'blocks.11.mlp.fc2.bias', 'norm.weight', 'norm.bias', 'head.weight', 'head.bias']
# }
# save_config(training_config)
#training_config = load_config("michelle_diet_imagenetmean_augmentD__freeze10__explr_1en4_0.8__AdamW_wdecay_1en4")
training_config = load_config("freeze10_explr_1en4_0.8_static_pseudolabeled_3AA_tomato_ext")

training_config

def save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler=None):
    checkpoint_dict = {
        "epoch": epoch,
        "model_dict": model.state_dict(),
        "optimizer_dict": optimizer.state_dict(),
    }

    # Consistently name the scheduler key as "scheduler_dict"
    if scheduler:
        checkpoint_dict["scheduler_dict"] = scheduler.state_dict()

    torch.save(checkpoint_dict, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_dict"])

    # Consistently access the scheduler as "scheduler_dict"
    if scheduler and "scheduler_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_dict"])

    return checkpoint["epoch"]

class TrainingHistory:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def save(self, history_path):
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "train_acc": self.train_acc,
                "val_acc": self.val_acc,
            }, f)

    def load(self, history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            hist = json.load(f)
        self.train_loss = hist["train_loss"]
        self.val_loss = hist["val_loss"]
        self.train_acc = hist["train_acc"]
        self.val_acc = hist["val_acc"]

    def append(self, train_loss, val_loss, train_acc, val_acc):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)

    def is_best(self):
        """Return true if the last epoch added is the best seen so far"""
        return all([self.val_loss[-1] < i for i in self.val_loss[:-1]])

class ImageDataset(Dataset):
    def __init__(self, root_dir, label_csv=None, return_filenames=False, transform=None):
        self.label_values = None
        self.return_filenames = return_filenames
        if label_csv is not None:
            csv_data = pd.read_csv(label_csv)
            self.filenames = csv_data["image"].tolist()
            self.label_values = csv_data["id"].tolist()
        else:
            self.filenames = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        idx = int(idx)
        img_name = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        result = [image]

        if self.label_values is not None:
            result.append(self.label_values[idx])

        if self.return_filenames:
            result.append(self.filenames[idx])

        return result

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))  # Adjust mean and std as needed
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

if not "augment_mode" in training_config:
    augmented_transform = transform
else:
    if training_config["augment_mode"] == "augment":
        # Data augmentation https://www.kaggle.com/code/pdochannel/vision-transformers-in-pytorch-deit/notebook
        augmented_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.25),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                    transforms.RandomErasing(p=0.2, value='random')
                ])
    elif training_config["augment_mode"] == "augmentD":
        # Less augmented
        augmented_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.25),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ])
    else:
        raise RuntimeError("Unknown augmentation")
    augmented_transform


# 3Augment args setup
class aug3Args:
    def __init__(self, input_size, src, jitter):
        self.input_size = input_size  # defaults to 224 in their code
        self.src = src  # "simple random crop" = True or False, False does a bit more
        self.color_jitter = jitter  # percentage, defaults to 0.3
AA3 = aug3Args(224, False, 0.3)

ignore = """
# Initialize datasets
train_dataset = ImageDataset(os.path.join(data_prefix,'train/labeled'), label_csv=os.path.join(data_prefix,'train_labeled.csv'), transform=augmented_transform)
val_dataset = ImageDataset(os.path.join(data_prefix,'train/labeled'), label_csv=os.path.join(data_prefix,'train_labeled.csv'), transform=transform)
# unlabeled_dataset = ImageDataset(os.path.join(data_prefix,'train/unlabeled'), transform=augmented_transform)

# The unlabeled dataset is notably larger - let's not augment it just yet
unlabeled_dataset = ImageDataset(os.path.join(data_prefix,'train/unlabeled'), transform=transform)
"""
train_dataset = ImageDataset(os.path.join(data_prefix,'train/labeled'), label_csv=os.path.join(data_prefix,'train_labeled.csv'), transform=new_data_aug_generator(AA3))
val_dataset = ImageDataset(os.path.join(data_prefix,'train/labeled'), label_csv=os.path.join(data_prefix,'train_labeled.csv'), transform=transform)
unlabeled_dataset = ImageDataset(os.path.join(data_prefix,'train/unlabeled'), label_csv=os.path.join(data_prefix,'unlabeled_training_labels.csv'), transform=new_data_aug_generator(AA3))

tomato_train_dataset = ImageDataset(os.path.join(tomato_prefix,'train/'), label_csv=os.path.join(data_prefix,'tomato.csv'), transform=new_data_aug_generator(AA3))

# Training / validation split
val_ratio = 0.1
batch_size = 16

val_size = int(val_ratio * len(train_dataset))
train_size = len(train_dataset) - val_size

generator1 = torch.Generator().manual_seed(12341234)
# Generate as indices so we can save them if needed, but I'm not doing that yet - Daniel
val_idx, train_idx = torch.utils.data.random_split(torch.arange(len(train_dataset)), [val_size, train_size], generator=generator1)
base_train =  torch.utils.data.Subset(train_dataset, train_idx)
val =  torch.utils.data.Subset(val_dataset, val_idx)

train = torch.utils.data.ConcatDataset([base_train,tomato_train_dataset])

labeled_train_data = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
labeled_val_data = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)

# a dataloader for the unlabeled data.  Batching at 100 to try to offset the effects of individual bad labels
unlabeled_batch_size = 100
unlabeled_data = DataLoader(unlabeled_dataset, batch_size=unlabeled_batch_size, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)

# # TODO: Ensure train/val is well split along class lines
# train_classes = [i[1] for i in train]
# val_classes = [i[1] for i in val]

# print(torch.unique(torch.as_tensor(train_classes),return_counts=True))
# print(torch.unique(torch.as_tensor(val_classes),return_counts=True))

# torch.unique(torch.as_tensor(train_classes)) == torch.unique(torch.as_tensor(val_classes))

# Function to display a batch of labeled images with labels
def show_labeled_batch(data_loader):
    images, labels = next(iter(data_loader))
    plt.figure(figsize=(12, 6))
    for idx in range(min(8, len(images))):
        plt.subplot(2, 4, idx + 1)
        img = images[idx].permute(1, 2, 0) # Convert from Tensor format
        img = img/2 + 0.5 # This roughly un-normalizes them back to a valid range for imshow - Daniel
        plt.imshow(img)
        plt.title(f'Label: {labels[idx].item()}')
        plt.axis('off')
    plt.show()

# Function to display a batch of unlabeled images
def show_unlabeled_batch(data_loader):
    images = next(iter(data_loader))
    plt.figure(figsize=(12, 6))
    for idx in range(min(8, len(images))):
        plt.subplot(2, 4, idx + 1)
        img = images[idx].permute(1, 2, 0) # Convert from Tensor format
        img = img/2 + 0.5 # This roughly un-normalizes them back to a valid range for imshow - Daniel
        plt.imshow(img)
        plt.title("Unlabeled Image")
        plt.axis('off')
    plt.show()

# Display a batch of labeled images
# skipping the display of labeled images for now
#print("Labeled images:")
#show_labeled_batch(labeled_train_data)
pass

# # Display a batch of unlabeled images
# print("Unlabeled images:")
# show_unlabeled_batch(unlabeled_loader)

# https://pytorch.org/tutorials/beginner/vt_tutorial.html
model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modify the final layer to match the number of classes
num_classes = len(categories)  # Adjust to the actual number of classes
model.head = nn.Linear(model.head.in_features, num_classes)

# https://www.kaggle.com/code/pdochannel/vision-transformers-in-pytorch-deit/notebook
criterion = LabelSmoothingCrossEntropy()
# criterion = criterion.to(device)
if not "optimizer_type" in training_config or training_config["optimizer_type"] == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=training_config["optimizer_lr"], weight_decay=training_config["optimizer_wd"])
elif training_config["optimizer_type"] == "AdamW":
    optimizer = optim.AdamW(model.parameters(), lr=training_config["optimizer_lr"], weight_decay=training_config["optimizer_wd"])
else:
    raise NotImplementedError()
print(optimizer)

def freeze_by_list(model, unfrozen):
    # https://stackoverflow.com/questions/62523912/freeze-certain-layers-of-an-existing-model-in-pytorch
    total_unfrozen = 0
    for name, param in model.named_parameters():
        if name in unfrozen:
            total_unfrozen += 1
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

if "unfreeze_layers" in training_config:
    freeze_by_list(model, training_config["unfreeze_layers"])

[(i[0], i[1].requires_grad) for i in model.named_parameters()]

# lr scheduler
if "scheduler_type" in training_config:
    if training_config["scheduler_type"] == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **training_config["scheduler_params"])
    if training_config["scheduler_type"] == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **training_config["scheduler_params"])
    if training_config["scheduler_type"] == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **training_config["scheduler_params"])
else:
    lr_scheduler = None
print(lr_scheduler)

model_path = os.path.join(model_prefix, training_config["model_name"])
print(f"model path: {model_path}")
os.makedirs(model_path, exist_ok=True)

num_epochs = 20  # Adjust as needed

model.to(device)
start_epoch = 0
hist = TrainingHistory()

checkpoint_path = os.path.join(model_path, "checkpoint.pth")
if os.path.exists(checkpoint_path):
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, device=device)
    hist.load(os.path.join(model_path, "history.json"))

for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    model.train()
    for images, labels in tqdm.tqdm(labeled_train_data, desc=f"Train ({epoch+1}/{num_epochs})"):
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track statistics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()


    # Start using unlabeled data immediately, since we have static labels for it
    if True: #(epoch > 3):
        for images, labels in tqdm.tqdm(unlabeled_data, desc=f"Train (pseudo-labeled) ({epoch+1}/{num_epochs})"):
            #images = image_arr[0]
            #images = images.to(device)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # let's reduce the loss by a significant factor, since we're less sure of these labels
            #loss = loss * 0.00001
            loss.backward()
            optimizer.step()

            ignore = """
            # Track statistics
            u_train_loss += loss.item()
            _, u_predicted = outputs.max(1)
            u_train_total += labels.size(0)
            u_train_correct += u_predicted.eq(labels).sum().item()
    else:
        u_train_total += 1 """


    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    for images, labels in tqdm.tqdm(labeled_val_data, desc=f"Validation ({epoch+1}/{num_epochs})"):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Track statistics
        val_loss += loss.item()
        _, predicted = outputs.max(1)
        val_total += labels.size(0)
        val_correct += predicted.eq(labels).sum().item()

    if training_config["scheduler_type"] == "ReduceLROnPlateau":
        lr_scheduler.step(val_loss)
    elif lr_scheduler:
        lr_scheduler.step()

    tmp_checkpoint_path = os.path.join(model_path, f"checkpoint-{epoch}.pth")
    save_checkpoint(tmp_checkpoint_path, epoch + 1, model, optimizer, lr_scheduler)
    shutil.copyfile(tmp_checkpoint_path, os.path.join(model_path, f"checkpoint.pth"))
    if hist.is_best():
        shutil.copyfile(tmp_checkpoint_path, os.path.join(model_path, f"checkpoint-best.pth"))
    os.unlink(tmp_checkpoint_path)

    hist.append(train_loss/train_total, val_loss/val_total, train_correct/train_total, val_correct/val_total)
    hist.save(os.path.join(model_path, "history.json"))

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/train_total:.6f}, Val Loss: {val_loss/val_total:.6f}")
    print(f"      Train Accuracy: {100 * train_correct/train_total:.2f}%, Val Accuracy: {100 * val_correct/val_total:.2f}%")
    print(f"      New LR={[g['lr'] for g in optimizer.param_groups]}")

# Plot Loss
plt.plot(range(len(hist.train_loss)), hist.train_loss, label="Train")
plt.plot(range(len(hist.val_loss)), hist.val_loss, label="Validation")
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Epoch")
#plt.show()

# Plot Accuracy
plt.plot(range(len(hist.train_acc)), hist.train_acc, label="Train")
plt.plot(range(len(hist.val_acc)), hist.val_acc, label="Validation")
plt.legend()
plt.ylabel("% Accuracy")
plt.xlabel("Epoch")
#plt.show()

#raise "STOP"

test_dataset = ImageDataset(root_dir=os.path.join(data_prefix,'test'), return_filenames=True, transform=transform)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)

import csv

def create_csv_with_number(base_name, data):
    """Creates a CSV file with a unique number in the filename if the file already exists."""

    file_number = 1
    file_name = f"{base_name}.csv"

    while os.path.exists(file_name):
        file_name = f"{base_name}_{file_number}.csv"
        file_number += 1

    # Save to CSV
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image", "id"])  # Write header
        writer.writerows(data)  # Write predictions

    print(f"File '{file_name}' created and results saved successfully.")

hist = None
best_checkpoint_path = os.path.join(model_path, "checkpoint-best.pth")
best_epoch = load_checkpoint(best_checkpoint_path, model, optimizer, lr_scheduler, device=device)
print(best_checkpoint_path)
print("Best epoch:", best_epoch)

model.to(device)
model.eval()

results = []
if True:
    with torch.no_grad():
        for images, filenames in tqdm.tqdm(test_data, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)  # Get the predicted class IDs

            # Store filename and predicted label
            results.extend(zip(filenames, predicted.cpu().numpy()))

    create_csv_with_number("test_submission", results)
    results = []

