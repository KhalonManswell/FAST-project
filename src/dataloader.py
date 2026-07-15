from pathlib import Path

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#Dataset paths
data_dir = Path("data/processed")

train_dir = data_dir / "train"
val_dir = data_dir / "val"
test_dir = data_dir / "test"

#Applying Transformations
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

evaluation_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


#Image Folder dataset creation:
train_dataset = datasets.ImageFolder(
    root=train_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=val_dir,
    transform=evaluation_transform
)

test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=evaluation_transform
)


#Dataloaders
batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)




