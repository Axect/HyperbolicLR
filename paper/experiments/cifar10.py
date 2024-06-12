import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from hyperbolic_lr import HyperbolicLR, ExpHyperbolicLR
import wandb
import random
import numpy as np

# Load CIFAR-10 dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 학습 함수
def train(model, optimizer, scheduler, num_epochs):
    wandb.watch(model, log="all")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        scheduler.step()

        train_loss /= len(trainloader)
        val_loss /= len(testloader)
        lr = scheduler.get_last_lr()[0]

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    optimizers = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
    }

    optimizer_params = {
        "SGD": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
        "Adam": {"lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 5e-4},
    }

    schedulers = {
        "PolynomialLR": optim.lr_scheduler.PolynomialLR,
        "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
        "HyperbolicLR": HyperbolicLR,
        "ExpHyperbolicLR": ExpHyperbolicLR,
    }

    num_epochs = 100

    scheduler_params = {
        "PolynomialLR": {"power": 0.5, "total_iters": num_epochs},
        "CosineAnnealingLR": {"T_max": num_epochs, "eta_min": 1e-4},
        "ExponentialLR": {"gamma": 0.95},
        "HyperbolicLR": {"upper_bound": 250, "max_iter": num_epochs, "init_lr": 0.1, "infimum_lr": 1e-4},
        "ExpHyperbolicLR": {"upper_bound": 250, "max_iter": num_epochs, "init_lr": 0.1, "infimum_lr": 1e-4},
    }

    for optimizer_name, optimizer_class in optimizers.items():
        for scheduler_name, scheduler_class in schedulers.items():
            # Fix seed for reproducibility
            seed = 71 # from random.org
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(seed)

            run_name = f"{optimizer_name}_{scheduler_name}"

            model = models.resnet18(pretrained=False, num_classes=10)
            net = model.to(device)

            wandb.init(project="cifar10-classification", name=run_name)

            optimizer = optimizer_class(net.parameters(), **optimizer_params[optimizer_name])
            scheduler = scheduler_class(optimizer, **scheduler_params[scheduler_name])

            train(net, optimizer, scheduler, num_epochs)
            wandb.finish()
