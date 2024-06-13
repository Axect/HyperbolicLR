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
import survey

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
testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Train function
def train(model, optimizer, scheduler, num_epochs, device):
    wandb.watch(model, log="all")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for (inputs, labels) in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if scheduler is not None:
            scheduler.step()

        train_loss /= len(trainloader)
        val_loss /= len(testloader)
        test_acc = 100 * correct / total
        lr = optimizer.param_groups[0]['lr']

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr, "test_acc": test_acc})
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}, Test Acc: {test_acc:.2f}")


# Main
if __name__ == "__main__":
    # Device
    device_count = torch.cuda.device_count()
    if device_count > 1:
        options = [f"cuda:{i}" for i in range(device_count)] + ["cpu"]
        device = survey.routines.select(
            "Select device",
            options=options
        )
        device = options[device]
    elif device_count == 1:
        device = "cuda:0"
    else:
        device = "cpu"
    print(device)

    criterion = nn.CrossEntropyLoss()

    optimizers = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
    }

    optimizer_params = {
        "SGD": {"lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4},
        "Adam": {"lr": 1e-4},
        "AdamW": {"lr": 1e-4, "betas": (0.85, 0.98)},
    }

    schedulers = {
        "PolynomialLR": optim.lr_scheduler.PolynomialLR,
        "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
        "ExponentialLR": optim.lr_scheduler.ExponentialLR,
        "HyperbolicLR": HyperbolicLR,
        "ExpHyperbolicLR": ExpHyperbolicLR,
    }

    num_epochs = 50

    scheduler_params = {
        "PolynomialLR": {"power": 0.5, "total_iters": num_epochs},
        "CosineAnnealingLR": {"T_max": num_epochs, "eta_min": 1e-4},
        "ExponentialLR": {"gamma": 0.9},
        "HyperbolicLR": {"upper_bound": 150, "max_iter": num_epochs, "init_lr": 0.01, "infimum_lr": 1e-4},
        "ExpHyperbolicLR": {"upper_bound": 150, "max_iter": num_epochs, "init_lr": 0.01, "infimum_lr": 1e-4},
    }

    def adjust_params_for_adam(scheduler_name, params, optimizer_name):
        if optimizer_name in ["Adam", "AdamW"]:
            adjusted_params = params.copy()
            if scheduler_name in ["HyperbolicLR", "ExpHyperbolicLR"]:
                adjusted_params["init_lr"] = params["init_lr"] / 100
                adjusted_params["infimum_lr"] = params["infimum_lr"] / 100
            elif scheduler_name == "CosineAnnealingLR":
                adjusted_params["eta_min"] = params["eta_min"] / 100
            return adjusted_params
        return params

    # Training without LR scheduler
    for optimizer_name, optimizer_class in optimizers.items():
        # Fix seed for reproducibility
        seed = 71 # from random.org
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

        run_name = f"{optimizer_name}_NoScheduler"

        model = models.vgg16(pretrained=False, num_classes=10)
        net = model.to(device)

        wandb.init(project="cifar10-classification", name=run_name)

        optimizer = optimizer_class(net.parameters(), **optimizer_params[optimizer_name])

        train(net, optimizer, None, num_epochs, device=device)
        wandb.finish()

    # Training with LR scheduler
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

            model = models.vgg16(pretrained=False, num_classes=10)
            net = model.to(device)

            wandb.init(project="cifar10-classification", name=run_name)

            optimizer = optimizer_class(net.parameters(), **optimizer_params[optimizer_name])
            scheduler_param = adjust_params_for_adam(scheduler_name, scheduler_params[scheduler_name], optimizer_name)

            scheduler = scheduler_class(optimizer, **scheduler_param)

            train(net, optimizer, scheduler, num_epochs, device=device)
            wandb.finish()

