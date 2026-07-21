import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


#Dataset paths
train_data_path = "data/processed/train"
test_data_path = "data/processed/test"
val_data_path = "data/processed/val"

#Data transformations and augmentations
train_transform = transforms.Compose([
    transforms.Resize(64),

    transforms.RandomAffine(
        degrees=5, 
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),

    transforms.ColorJitter(brightness=0.10, contrast=0.10),
    transforms.ToTensor(),
])

evaluation_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()
])

#Image folder creation
train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=evaluation_transform)
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=evaluation_transform)

#Dataloaders
batch_size = 32
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size)
val_data_loader = DataLoader(val_data, batch_size=batch_size)