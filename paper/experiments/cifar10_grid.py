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

class CNNHyperparameterSearch:
    def __init__(self, batch_size, epochs, lr, infimum_lr):
        self.hparams = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "infimum_lr": infimum_lr,
        }
        self.cnn_param_space = {
            "num_conv_layers": [3, 4],
            "num_fc_layers": [2, 3, 4],
            "conv_channels": [32, 64, 128],
            "fc_units": [128, 256, 512]
        }
        self.schedulers = {
            "PolynomialLR": optim.lr_scheduler.PolynomialLR,
            "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
            "ExponentialLR": optim.lr_scheduler.ExponentialLR,
            "HyperbolicLR": HyperbolicLR,
            "ExpHyperbolicLR": ExpHyperbolicLR,
        }
        self.scheduler_params = {
            "PolynomialLR": {"power": 0.9},
            "CosineAnnealingLR": {"eta_min": infimum_lr},
            "ExponentialLR": {"gamma": 0.95},
            "HyperbolicLR": {"upper_bound": 250, "infimum_lr": infimum_lr},
            "ExpHyperbolicLR": {"upper_bound": 250, "infimum_lr": infimum_lr},
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

    def train_and_evaluate(self, model, optimizer, scheduler, epochs, device, run):
        criterion = nn.CrossEntropyLoss()
        model.to(device)
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(self.trainloader)
            
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in self.testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss /= len(self.testloader)
            accuracy = correct / total
            
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
            
            # Log metrics
            run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "learning_rate": current_lr
            })
            
            scheduler.step()
        
        return accuracy

    def grid_search(self, scheduler_name, epochs_list, seeds, device, project_name):
        results = {}
        progress = Progress()
        
        with progress:
            total_progress = progress.add_task(f"[green]Searching with {scheduler_name}", total=len(epochs_list))
            
            for epochs in epochs_list:
                self.hparams["epochs"] = epochs
                best_accuracy = 0
                best_params = None
                
                param_progress = progress.add_task(f"[blue]CNN parameters for {epochs} epochs", total=len(list(itertools.product(*self.cnn_param_space.values()))))
                
                for params in itertools.product(*self.cnn_param_space.values()):
                    param_dict = dict(zip(self.cnn_param_space.keys(), params))
                    accuracies = []
                    
                    # Create a shortened name for the CNN configuration
                    cnn_config = f"C{param_dict['num_conv_layers']}F{param_dict['num_fc_layers']}_Ch{param_dict['conv_channels']}_FC{param_dict['fc_units']}"
                    
                    seed_progress = progress.add_task(f"[cyan]Seeds", total=len(seeds))
                    
                    for seed in seeds:
                        random.seed(seed)
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        torch.backends.cudnn.deterministic = True
                        torch.cuda.manual_seed_all(seed)
    
                        model = SimpleCNN(**param_dict)
                        optimizer = optim.AdamW(model.parameters(), lr=self.hparams["learning_rate"])
                        
                        scheduler_params = self.scheduler_params[scheduler_name].copy()
                        if scheduler_name in ["HyperbolicLR", "ExpHyperbolicLR"]:
                            scheduler_params["max_iter"] = epochs
                            scheduler_params["init_lr"] = self.hparams["learning_rate"]
                        elif scheduler_name == "PolynomialLR":
                            scheduler_params["total_iters"] = epochs
                        elif scheduler_name == "CosineAnnealingLR":
                            scheduler_params["T_max"] = epochs
                        
                        scheduler = self.schedulers[scheduler_name](optimizer, **scheduler_params)
                        
                        run = wandb.init(
                            project=project_name,
                            config={"scheduler": scheduler_name, "epochs": epochs, **param_dict, **self.hparams},
                            name=f"{scheduler_name}_{epochs}ep_{cnn_config}_seed{seed}",
                            group=f"{scheduler_name}_{epochs}ep_{cnn_config}"
                        )
                        
                        accuracy = self.train_and_evaluate(model, optimizer, scheduler, epochs, device, run)
                        accuracies.append(accuracy)
                        
                        run.finish()
                        
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
        epochs_list = [50, 100, 150]
        
        for scheduler_name in self.schedulers.keys():
            results = self.grid_search(scheduler_name, epochs_list, seeds, device, project_name)
            
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

    project_name = "CNNHyperparameterSearch"

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

    search_model = CNNHyperparameterSearch(batch_size, epochs, learning_rate, infimum_lr)
    search_model.search(seeds[:3], project_name, device)

if __name__ == "__main__":
    main()
