import torch
import torch.nn as nn

from datasets import train_data_loader

class BaselineCNN(nn.Module):

    def __init__(self, num_classes = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1 ,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1 ,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d( (1,1) )
    
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )



    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

images, labels = next(iter(train_data_loader))

model = BaselineCNN()
outputs = model(images)
print(images.shape)
print(outputs.shape)
