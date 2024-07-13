import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, Subset
import torchvision
import torchvision.transforms as transforms
import wandb

import polars as pl
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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


# Load Data
def load_osc_data(dtype="simple", mode="S", hist=10, pred=10, ratio=0.8):
    """
    dtype:
        - "simple": Simple Harmonic Oscillator
        - "damped": Damped Harmonic Oscillator 

    mode:
        - "S": Single input, Single target ('x')
        - "M": Multi input, Multi target
        - "MS": Multi input, Single target ('x')

    Columns:
        - t: Time (Not used)
        - x: Position
        - v: Velocity
        - a: Acceleration
        - zeta: Damping parameter (0, 0.1, 0.2)
    """
    df = pl.read_parquet("./data/damped_sho.parquet")
    df = df.drop("t")

    # Simple or Damped or Total
    if dtype == "simple":
        df = df.filter(pl.col('zeta') == 0)
    elif dtype == "damped":
        df = df.filter(pl.col('zeta') == 0.01)
    elif dtype == "total":
        df1 = df.filter(pl.col('zeta') == 0)
        df2 = df.filter(pl.col('zeta') == 0.01)
        df3 = df.filter(pl.col('zeta') == 0.02)
        df = pl.concat([df1, df2, df3])
    else:
        raise ValueError("dtype must be 'simple' or 'damped'")

    # Normalize columns
    #df_min = df.min()
    #df_max = df.max()

    #def normalize(col):
    #    return (col - df_min[col.name]) / (df_max[col.name] - df_min[col.name])

    #df = df.with_columns([
    #    pl.all().map(normalize)
    #])
    df = df.select([
        ((pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min())).alias(col)
        for col in df.columns
    ])

    if mode == 'S':
        if dtype == "total":
            df1 = df.filter(pl.col('zeta') == 0)
            df2 = df.filter(pl.col('zeta') == 0.01)
            df3 = df.filter(pl.col('zeta') == 0.02)
            x1  = df1['x'].to_numpy()
            x2  = df2['x'].to_numpy()
            x3  = df3['x'].to_numpy()
            x1_slide = sliding_window_view(x1, hist + pred)
            x2_slide = sliding_window_view(x2, hist + pred)
            x3_slide = sliding_window_view(x3, hist + pred)
            x1_hist  = x1_slide[:, :hist]
            x2_hist  = x2_slide[:, :hist]
            x3_hist  = x3_slide[:, :hist]
            x1_pred  = x1_slide[:, -pred:]
            x2_pred  = x2_slide[:, -pred:]
            x3_pred  = x3_slide[:, -pred:]
            input1 = torch.tensor(x1_hist, dtype=torch.float32).unsqueeze(2)
            input2 = torch.tensor(x2_hist, dtype=torch.float32).unsqueeze(2)
            input3 = torch.tensor(x3_hist, dtype=torch.float32).unsqueeze(2)
            label1 = torch.tensor(x1_pred, dtype=torch.float32).unsqueeze(2)
            label2 = torch.tensor(x2_pred, dtype=torch.float32).unsqueeze(2)
            label3 = torch.tensor(x3_pred, dtype=torch.float32).unsqueeze(2)
            input_data = torch.cat([input1, input2, input3], dim=0)
            label_data = torch.cat([label1, label2, label3], dim=0)
        else:
            x = df['x'].to_numpy()

            # Sliding window
            x_slide = sliding_window_view(x, hist + pred)   # M x (hist + pred)
            x_hist  = x_slide[:, :hist]                     # M x hist
            x_pred  = x_slide[:, -pred:]                    # M x pred

            input_data = torch.tensor(x_hist, dtype=torch.float32).unsqueeze(2)
            label_data = torch.tensor(x_pred, dtype=torch.float32).unsqueeze(2)
    elif mode == 'MS' or mode == 'M':
        x = df['x'].to_numpy()
        v = df['v'].to_numpy()
        a = df['a'].to_numpy()

        # Sliding window
        x_slide = sliding_window_view(x, hist + pred)   # N x (hist + pred)
        v_slide = sliding_window_view(v, hist + pred)   # N x (hist + pred)
        a_slide = sliding_window_view(a, hist + pred)   # N x (hist + pred)
        x_hist  = x_slide[:, :hist]                     # N x hist
        v_hist  = v_slide[:, :hist]                     # N x hist
        a_hist  = a_slide[:, :hist]                     # N x hist
        x_pred  = x_slide[:, -pred:]                    # N x pred
        v_pred  = v_slide[:, -pred:]                    # N x pred
        a_pred  = a_slide[:, -pred:]                    # N x pred
        x_hist = np.expand_dims(x_hist, axis=2)         # N x hist x 1
        v_hist = np.expand_dims(v_hist, axis=2)         # N x hist x 1
        a_hist = np.expand_dims(a_hist, axis=2)         # N x hist x 1
        
        input_array = np.concatenate([x_hist, v_hist, a_hist], axis=2) # N x hist x 3
        input_data = torch.tensor(input_array, dtype=torch.float32)

        if mode == 'MS':
            label_data = torch.tensor(x_pred, dtype=torch.float32)
            label_data = label_data.unsqueeze(2)
        elif mode == 'M':
            x_pred = np.expand_dims(x_pred, axis=2)
            v_pred = np.expand_dims(v_pred, axis=2)
            a_pred = np.expand_dims(a_pred, axis=2)
            label_array = np.concatenate([x_pred, v_pred, a_pred], axis=2) # N x pred x 3
            label_data = torch.tensor(label_array, dtype=torch.float32)
        else:
            raise ValueError("mode must be 'MS' or 'M'")
    else:
        raise ValueError("mode must be 'S' or 'M'")

    print(f"Input shape: {input_data.shape}")
    print(f"Label shape: {label_data.shape}")

    ds = TensorDataset(input_data, label_data)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    ds_train, ds_val = random_split(ds, [train_size, val_size])
    return ds_train, ds_val


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
