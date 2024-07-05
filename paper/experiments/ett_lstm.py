import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from hyperbolic_lr import HyperbolicLR, ExpHyperbolicLR
from rich.progress import Progress
from itertools import product

import wandb
import survey
import random
import os
import json
import math


def load_ett_data(mode="train"):
    """
    Load the ETT dataset

    Args:
        mode (str): Either 'train' or 'test'

    Columns:
        - 'group': group number of the time series (8 * 24 = 192 lists in a group)
        - 'type' : whether it is input or label (input = 0, label = 1; input: 1 * 24, label: 1 * 24)
        - 'HUFL' : High UseFul Load     (input)
        - 'HULL' : High UseLess Load    (input)
        - 'MUFL' : Middle UseFul Load   (input)
        - 'MULL' : Middle UseLess Load  (input)
        - 'LUFL' : Low UseFul Load      (input)
        - 'LULL' : Low UseLess Load     (input)
        - 'OT'   : Oil Temperature      (target)
    """
    if mode == "train":
        df = pd.read_parquet("./data/ETTh1_train.parquet")
    elif mode == "test":
        df = pd.read_parquet("./data/ETTh1_test.parquet")
    else:
        raise ValueError("mode must be either 'train' or 'test'")

    # Separate input and label data
    input_data = df[df['type'] == 0][['group', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']]
    label_data = df[df['type'] == 1][['group', 'OT']]
    
    # Group the data
    grouped_input = input_data.groupby('group')
    grouped_label = label_data.groupby('group')
    
    # Convert to list of tensors
    input_tensors = [torch.tensor(group[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].values, dtype=torch.float32) 
                     for _, group in grouped_input]
    label_tensors = [torch.tensor(group['OT'].values, dtype=torch.float32) 
                     for _, group in grouped_label]
    
    # Stack tensors
    inputs = torch.stack(input_tensors)
    labels = torch.stack(label_tensors).unsqueeze(2)

    print(f"Input shape: {inputs.shape}")
    print(f"Label shape: {labels.shape}")
    
    # Create TensorDataset
    dataset = TensorDataset(inputs, labels)
    
    return dataset


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell


class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.decoder = Decoder(output_dim, hidden_dim, output_dim, num_layers)

    def forward(self, src, target, teacher_forcing_ratio=0.5):
        batch_size, seq_len, _ = src.shape
        
        # Encode the input sequence
        hidden, cell = self.encoder(src)
        
        # Initialize decoder input
        decoder_input = torch.zeros(batch_size, 1, 1, device=src.device)
        
        # Initialize list to store predictions
        outputs = torch.zeros(batch_size, 24, 1, device=src.device)
        
        # Decode one step at a time
        for t in range(24):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t:t+1] = output
            
            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = target[:, t:t+1] if teacher_force else output
        
        return outputs


class Trainer:
    def __init__(self, model, optimizer, scheduler, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model.to(self.device)

    def step(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)
        output = self.model(input, target)
        return output

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0
        for input, label in dataloader:
            input = input.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(input, label)
            loss = F.mse_loss(pred, label)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        return epoch_loss

    def evaluate(self, dataloader):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for input, label in dataloader:
                input = input.to(self.device)
                label = label.to(self.device)
                pred = self.model(input, label)
                loss = F.mse_loss(pred, label)
                epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        return epoch_loss

    def train(self, dl_train, dl_val, progress, epochs):
        progress_epoch = progress.add_task("[cyan]Epochs", total=epochs)
        for epoch in range(epochs):
            train_loss = self.train_epoch(dl_train)
            val_loss = self.evaluate(dl_val)
            if self.scheduler is not None:
                self.scheduler.step()
            progress.update(
                progress_epoch, 
                advance=1,
                description=f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}"
            )
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch": epoch+1,
            })
        progress.remove_task(progress_epoch)

def main():
    # Device selection
    device_count = torch.cuda.device_count()
    if device_count > 1:
        options = [f"cuda:{i}" for i in range(device_count)] + ["cpu"]
        device_index = survey.routines.select(
            "Select device",
            options=options
        )
        device = options[device_index]
    elif device_count == 1:
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Hyperparameters
    hidden_dim = survey.routines.numeric("Hidden dimension", decimal=False)
    num_layers = survey.routines.numeric("Number of layers", decimal=False)
    batch_size = survey.routines.numeric("Batch size", decimal=False)
    num_epochs = survey.routines.numeric("Number of epochs", decimal=False)
    lr_no_scheduler = survey.routines.numeric("Learning rate (without scheduler; for Adam/AdamW)")
    lr_scheduler = survey.routines.numeric("Learning rate (with scheduler; for Adam/AdamW)")
    infimum_lr = survey.routines.numeric("Infimum learning rate")

    hparams = {
        "input_dim": 7,  # HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
        "output_dim": 1,  # OT (Oil Temperature)
        "hidden_dim": 256,
        "num_layers": 2,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr_no_scheduler": lr_no_scheduler,
        "lr_scheduler": lr_scheduler,
        "infimum_lr": infimum_lr
    }

    # Data
    train_data = load_ett_data("train")
    val_data = load_ett_data("test")
    dl_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Optimizer & Scheduler
    optimizers = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
        "AdamW": optim.AdamW
    }
    optimizer_params_no_scheduler = {
        "SGD": {"lr": lr_no_scheduler * 10, "momentum": 0.9, "weight_decay": 5e-4},
        "Adam": {"lr": lr_no_scheduler},
        "AdamW": {"lr": lr_no_scheduler, "betas": (0.85, 0.98)},
    }
    optimizer_params = {
        "SGD": {"lr": lr_scheduler * 10, "momentum": 0.9, "weight_decay": 5e-4},
        "Adam": {"lr": lr_scheduler},
        "AdamW": {"lr": lr_scheduler, "betas": (0.85, 0.98)},
    }
    schedulers = {
        "PolynomialLR": optim.lr_scheduler.PolynomialLR,
        "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
        "ExponentialLR": optim.lr_scheduler.ExponentialLR,
        "HyperbolicLR": HyperbolicLR,
        "ExpHyperbolicLR": ExpHyperbolicLR,
    }
    scheduler_params = {
        "PolynomialLR": {"power": 0.5, "total_iters": num_epochs},
        "CosineAnnealingLR": {"T_max": num_epochs, "eta_min": infimum_lr},
        "ExponentialLR": {"gamma": 0.97},
        "HyperbolicLR": {"upper_bound": 250, "max_iter": num_epochs, "init_lr": lr_scheduler, "infimum_lr": infimum_lr},
        "ExpHyperbolicLR": {"upper_bound": 250, "max_iter": num_epochs, "init_lr": lr_scheduler, "infimum_lr": infimum_lr},
    }
    def adjust_params_for_SGD(scheduler_name, params, optimizer_name):
        if optimizer_name == "SGD":
            adjusted_params = params.copy()
            if scheduler_name == "CosineAnnealingLR":
                adjusted_params["eta_min"] = params["eta_min"] * 10
            elif scheduler_name in ["HyperbolicLR", "ExpHyperbolicLR"]:
                adjusted_params["init_lr"] = params["init_lr"] * 10
                adjusted_params["infimum_lr"] = params["infimum_lr"] * 10
            return adjusted_params
        return params
    # ──────────────────────────────────────────────────────────────────────
    # Training

    # Seeds from random.org
    seeds = [89, 231, 928, 814, 269]

    # Training without LR scheduler
    for optimizer_name, optimizer_class in optimizers.items():
        for seed in seeds:
            # Fix seed for reproducibility
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(seed)

            group_name = f"{optimizer_name}_NoScheduler({num_epochs})"
            run_name = f"{group_name}[{seed}]"
            tags = ["NoScheduler", optimizer_name, f"{num_epochs}"]

            model = EncoderDecoderLSTM(hparams["input_dim"], hparams["hidden_dim"], hparams["output_dim"], hparams["num_layers"])
            net = model.to(device)

            # Optimizer & Scheduler
            optimizer = optimizer_class(net.parameters(), **optimizer_params_no_scheduler[optimizer_name])

            wandb.init(project="HyperbolicLR-ETT-LSTM", name=run_name, group=group_name, config=hparams, tags=tags)
            progress = Progress()

            trainer = Trainer(model, optimizer, None, device=device)
            trainer.train(dl_train, dl_val, progress, num_epochs)
            wandb.finish()

    # Training with LR scheduler
    for optimizer_name, optimizer_class in optimizers.items():
        for scheduler_name, scheduler_class in schedulers.items():
            for seed in seeds:
                # Fix seed for reproducibility
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed_all(seed)

                group_name = f"{optimizer_name}_{scheduler_name}({num_epochs})"
                run_name = f"{group_name}[{seed}]"
                tags = [scheduler_name, optimizer_name, f"{num_epochs}"]

                model = EncoderDecoderLSTM(hparams["input_dim"], hparams["hidden_dim"], hparams["output_dim"], hparams["num_layers"])
                net = model.to(device)

                # Optimizer & Scheduler
                optimizer = optimizer_class(net.parameters(), **optimizer_params_no_scheduler[optimizer_name])
                scheduler_param = adjust_params_for_SGD(scheduler_name, scheduler_params[scheduler_name], optimizer_name)
                scheduler = scheduler_class(optimizer, **scheduler_param)

                wandb.init(project="HyperbolicLR-ETT-LSTM(Explore)", name=run_name, group=group_name, config=hparams, tags=tags)
                progress = Progress()

                trainer = Trainer(model, optimizer, scheduler, device=device)
                trainer.train(dl_train, dl_val, progress, num_epochs)
                wandb.finish()


if __name__ == "__main__":
    main()