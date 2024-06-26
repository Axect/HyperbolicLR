import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from hyperbolic_lr import HyperbolicLR, ExpHyperbolicLR
import wandb
import random
import numpy as np
import survey
from rich.progress import Progress
import itertools
import os
import json

class SimpleCNN(nn.Module):
    def __init__(self, num_conv_layers, num_fc_layers, conv_channels, fc_units):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        
        # Convolutional layers
        in_channels = 3
        for _ in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(2))
            in_channels = conv_channels
        
        # Fully connected layers
        fc_input = conv_channels * (32 // (2**num_conv_layers))**2
        for _ in range(num_fc_layers - 1):
            self.fc_layers.append(nn.Linear(fc_input, fc_units))
            self.fc_layers.append(nn.ReLU())
            fc_input = fc_units
        self.fc_layers.append(nn.Linear(fc_input, 10))
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers:
            x = layer(x)
        return x

class LRSchedulerSearch:
    def __init__(self, batch_size, epochs, lr, infimum_lr):
        self.hparams = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "infimum_lr": infimum_lr,
            "num_conv_layers": 3,
            "num_fc_layers": 2,
            "conv_channels": 64,
            "fc_units": 128
        }
        self.schedulers = {
            "PolynomialLR": optim.lr_scheduler.PolynomialLR,
            "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
            "ExponentialLR": optim.lr_scheduler.ExponentialLR,
            "HyperbolicLR": HyperbolicLR,
            "ExpHyperbolicLR": ExpHyperbolicLR,
        }
        self.param_space = {
            "PolynomialLR": {"power": [0.5, 1.0, 1.5]},
            "CosineAnnealingLR": {"eta_min": [1e-5, 1e-4, 1e-3]},
            "ExponentialLR": {"gamma": [0.9, 0.95, 0.99]},
            "HyperbolicLR": {"upper_bound": [100, 250, 500], "infimum_lr": [1e-5, 1e-4, 1e-3]},
            "ExpHyperbolicLR": {"upper_bound": [100, 250, 500], "infimum_lr": [1e-5, 1e-4, 1e-3]},
        }
        self.dataloader()

    def dataloader(self):
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
        self.trainloader = DataLoader(trainset, batch_size=self.hparams["batch_size"], shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = DataLoader(testset, batch_size=self.hparams["batch_size"], shuffle=False, num_workers=2)

    def train_and_evaluate(self, model, optimizer, scheduler, epochs, device):
        criterion = nn.CrossEntropyLoss()
        model.to(device)
        
        for epoch in range(epochs):
            model.train()
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total

    def grid_search(self, scheduler_name, param_space, epochs_list, seeds, device):
        results = {}
        progress = Progress()
        
        with progress:
            total_progress = progress.add_task(f"[green]Searching {scheduler_name}", total=len(epochs_list))
            
            for epochs in epochs_list:
                self.hparams["epochs"] = epochs
                best_accuracy = 0
                best_params = None
                
                param_progress = progress.add_task(f"[blue]Parameters for {epochs} epochs", total=len(list(itertools.product(*param_space.values()))))
                
                for params in itertools.product(*param_space.values()):
                    param_dict = dict(zip(param_space.keys(), params))
                    accuracies = []
                    
                    seed_progress = progress.add_task(f"[cyan]Seeds", total=len(seeds))
                    
                    for seed in seeds:
                        random.seed(seed)
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        torch.backends.cudnn.deterministic = True
                        torch.cuda.manual_seed_all(seed)

                        model = SimpleCNN(**{k: self.hparams[k] for k in ['num_conv_layers', 'num_fc_layers', 'conv_channels', 'fc_units']})
                        optimizer = optim.AdamW(model.parameters(), lr=self.hparams["learning_rate"])
                        
                        if scheduler_name in ["HyperbolicLR", "ExpHyperbolicLR"]:
                            param_dict["max_iter"] = epochs
                            param_dict["init_lr"] = self.hparams["learning_rate"]
                        elif scheduler_name == "PolynomialLR":
                            param_dict["total_iters"] = epochs
                        elif scheduler_name == "CosineAnnealingLR":
                            param_dict["T_max"] = epochs
                        
                        scheduler = self.schedulers[scheduler_name](optimizer, **param_dict)
                        
                        accuracy = self.train_and_evaluate(model, optimizer, scheduler, epochs, device)
                        accuracies.append(accuracy)
                        
                        progress.update(seed_progress, advance=1)
                    
                    progress.remove_task(seed_progress)
                    
                    avg_accuracy = sum(accuracies) / len(accuracies)
                    if avg_accuracy > best_accuracy:
                        best_accuracy = avg_accuracy
                        best_params = param_dict
                    
                    progress.update(param_progress, advance=1)
                
                progress.remove_task(param_progress)
                results[epochs] = (best_params, best_accuracy)
                progress.update(total_progress, advance=1)
        
        return results

    def search(self, seeds, project_name, device):
        epochs_list = [50, 100, 200]
        
        for scheduler_name, param_space in self.param_space.items():
            results = self.grid_search(scheduler_name, param_space, epochs_list, seeds, device)
            
            # Log results to wandb
            for epochs, (params, accuracy) in results.items():
                run = wandb.init(
                    project=project_name,
                    config={"scheduler": scheduler_name, "epochs": epochs, **params},
                    name=f"{scheduler_name}_{epochs}_epochs",
                    group=scheduler_name
                )
                wandb.log({"best_accuracy": accuracy})
                run.finish()
            
            # Save results locally
            if not os.path.exists(project_name):
                os.makedirs(project_name)
            
            result_path = os.path.join(project_name, f"{scheduler_name}_results.json")
            with open(result_path, "w") as f:
                json.dump(results, f)
            
            print(f"Results for {scheduler_name}:")
            for epochs, (params, accuracy) in results.items():
                print(f"  Epochs: {epochs}, Best params: {params}, Accuracy: {accuracy:.4f}")

def main():
    # Seeds from random.org
    seeds = [563, 351, 445, 688, 261]

    project_name = "CIFAR-LRSearch"

    # Define common hyperparameters
    batch_size = survey.routines.numeric(
        "Enter batch_size (e.g. 128)", decimal=False)
    epochs = survey.routines.numeric(
        "Enter maximum epochs (e.g. 200)", decimal=False)
    learning_rate = survey.routines.numeric(
        "Enter learning_rate (e.g. 1e-3)")
    infimum_lr = survey.routines.numeric(
        "Enter infimum_lr (e.g. 1e-6)")

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

    search_model = LRSchedulerSearch(batch_size, epochs, learning_rate, infimum_lr)
    search_model.search(seeds[:5], project_name, device)

if __name__ == "__main__":
    main()
