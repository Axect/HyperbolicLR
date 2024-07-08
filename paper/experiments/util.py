import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, Subset
import torchvision
import torchvision.transforms as transforms
import wandb

import numpy as np
import random


def load_cifar10(subset_ratio=0.1):
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
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Randomly choose subset of train data
    train_size = len(trainset)
    subset_size = int(train_size * subset_ratio)
    train_ics = np.random.choice(train_size, subset_size, replace=False)
    trainsubset = Subset(trainset, train_ics)

    # Randomly choose subset of test data
    test_size = len(testset)
    subset_size = int(test_size * subset_ratio)
    test_ics = np.random.choice(test_size, subset_size, replace=False)
    testsubset = Subset(testset, test_ics)

    trainset = trainsubset
    testset = testsubset

    return trainset, testset


def load_cifar100(subset_ratio=0.1):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    # Randomly choose subset of train data
    train_size = len(trainset)
    subset_size = int(train_size * subset_ratio)
    train_ics = np.random.choice(train_size, subset_size, replace=False)
    trainsubset = Subset(trainset, train_ics)
    # Randomly choose subset of test data
    test_size = len(testset)
    subset_size = int(test_size * subset_ratio)
    test_ics = np.random.choice(test_size, subset_size, replace=False)
    testsubset = Subset(testset, test_ics)
    trainset = trainsubset
    testset = testsubset
    return trainset, testset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion=F.mse_loss, acc=False, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = criterion
        self.acc = acc

    def step(self, x):
        pred = self.model(x.to(self.device))
        return pred

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0
        for x, y in dataloader:
            self.optimizer.zero_grad()
            pred = self.step(x)
            loss = self.criterion(pred, y.to(self.device))
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        return epoch_loss

    def evaluate(self, dataloader):
        self.model.eval()
        eval_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                pred = self.step(x)
                loss = self.criterion(pred, y.to(self.device))
                eval_loss += loss.item()
                if self.acc:
                    _, predicted = torch.max(pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y.to(self.device)).sum().item()
        eval_loss /= len(dataloader)
        if self.acc:
            accuracy = correct / total
            return eval_loss, accuracy
        else:
            return eval_loss

    def train(self, dl_train, dl_val, epochs=500):
        val_loss = 0.0
        accuracy = 0.0
        for epoch in range(epochs):
            train_loss = self.train_epoch(dl_train)
            if self.acc:
                val_loss, accuracy = self.evaluate(dl_val)
            else:
                val_loss = self.evaluate(dl_val)
            if self.scheduler is not None:
                self.scheduler.step()
            if self.acc:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch": epoch+1,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "acc": accuracy
                })
            else:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch": epoch+1,
                    "lr": self.optimizer.param_groups[0]['lr']
                })

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1:03}\ttrain_loss={train_loss:.4e}\tval_loss={val_loss:.4e}")

        if self.acc:
            return val_loss, accuracy
        else:
            return val_loss
