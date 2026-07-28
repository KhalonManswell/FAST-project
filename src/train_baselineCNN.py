import torch
import torch.nn as nn
import torch.optim as optim

import datasets

class BaselineCNN(nn.Module):

    def __init__(self, num_classes):
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


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs, device = 'cpu'):
    model = model.to(device)
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        valid_correct = 0

        model.train()

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            output = model(inputs)
            loss = loss_fn(output, targets)

            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                output = model(inputs)
                loss = loss_fn(output, targets)

                valid_loss += loss.item()

                predictions = output.argmax(dim = 1)
                valid_correct += (predictions == targets).sum().item()
                valid_accuracy = valid_correct / len(val_loader.dataset)

        print(
            f"Epoch: {epoch+1 }" f"/{epochs} | "
            f"Train Loss: {training_loss:.4f} | "
            f"Validation Loss: {valid_loss:.4f} | "
            f"Validation accuracy: {valid_accuracy:.2%} "
        )

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    return device


def main():
    device = get_device()

    model = BaselineCNN(num_classes=10)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train(model, optimizer, loss_fn, train_loader=datasets.train_data_loader, val_loader=datasets.val_data_loader, 
          epochs=20, device=device)
    

if __name__ == "__main__":
    main()