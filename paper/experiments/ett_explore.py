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


# ┌──────────────────────────────────────────────────────────┐
#  Load ETT dataset
# └──────────────────────────────────────────────────────────┘
def load_ett_data(mode="train"):
    """
    Load the ETT dataset

    Args:
        mode (str): Either 'train' or 'test'

    Columns:
        - 'group': group number of the time series (8 * 24 = 192 lists in a group)
        - 'type' : wheter it is input or label (input = 0, label = 1; input: 1 * 24, label: 1 * 24)
        - 'HUFL' : High UseFul Load     (input)
        - 'HULL' : High UseLess Load    (input)
        - 'MUFL' : Middle UseFul Load   (input)
        - 'MULL' : Middle UseLess Load  (input)
        - 'LUFL' : Low UseFul Load      (input)
        - 'LULL' : Low UseLess Load     (input)
        - 'OT'   : Oil Temperature      (target)

    Caution:
        - This dataset contains whole columns
        - You should filter the columns differently according to type
          - if type is input: use 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'
          - if type is label: use 'OT'
    """
    if mode == "train":
        df = pd.read_parquet("./data/ETTh1_train.parquet")
    elif mode == "test":
        df = pd.read_parquet("./data/ETTh1_test.parquet")
    else:
        raise ValueError("mode must be either 'train' or 'test'")

    # Separate input and label data
    input_data = df[df['type'] == 0][['group', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']]
    target_data = df[df['type'] == 0][['group', 'OT']]
    label_data = df[df['type'] == 1][['group', 'OT']]
    
    # Group the data
    grouped_input = input_data.groupby('group')
    grouped_target = target_data.groupby('group')
    grouped_label = label_data.groupby('group')
    
    # Convert to list of tensors
    input_tensors = [torch.tensor(group[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']].values, dtype=torch.float32) 
                     for _, group in grouped_input]
    target_tensors = [torch.tensor(group['OT'].values, dtype=torch.float32) 
                     for _, group in grouped_target]
    label_tensors = [torch.tensor(group['OT'].values, dtype=torch.float32) 
                     for _, group in grouped_label]
    
    ## Pad sequences to same length if necessay
    #max_len = max(len(t) for t in input_tensors)
    #input_tensors = [torch.nn.functional.pad(t, (0, 0, 0, max_len - len(t))) for t in input_tensors]
    #label_tensors = [torch.nn.functional.pad(t, (0, max_len - len(t))) for t in label_tensors]
    
    # Stack tensors
    inputs = torch.stack(input_tensors)
    targets = torch.stack(target_tensors).unsqueeze(2)
    labels = torch.stack(label_tensors).unsqueeze(2)

    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Label shape: {labels.shape}")
    
    # Create TensorDataset
    dataset = TensorDataset(inputs, targets, labels)
    
    return dataset


# ┌──────────────────────────────────────────────────────────┐
#  Network Model
# └──────────────────────────────────────────────────────────┘
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        - x: (B, W, d_model)
        - self.pe: (1, M, d_model)
        - self.pe[:, :x.size(1), :]: (1, W, d_model)
        - output: (B, W, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model, 
            nhead, 
            dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers,
            norm=nn.LayerNorm(d_model)
        )

    def forward(self, x):
        """
        - x: (B, W, input_dim)
        - x (after embedding): (B, W, d_model)
        - out: (B, W, d_model)
        """
        x = self.embedding(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        return out


class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(output_dim, d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model, 
            nhead, 
            dim_feedforward,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, 
            num_layers,
            norm=nn.LayerNorm(d_model)
        )
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x, memory):
        """
        - x: (B, W, output_dim)
        - x (after embedding): (B, W, d_model)
        - memory: (B, W, d_model)
        - out: (B, W, d_model)
        - out (after fc): (B, W, output_dim)
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_decoder(x)
        out = self.transformer_decoder(x, memory)
        out = self.fc(out)
        return out


class TFEncDec(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        d_model = hparams["d_model"]
        nhead   = hparams["nhead"]
        num_layers = hparams["num_layers"]
        dim_feedforward = hparams["dim_feedforward"]
        input_dim = hparams["input_dim"]
        output_dim = hparams["output_dim"]

        self.encoder = Encoder(input_dim, d_model, nhead, num_layers, dim_feedforward)
        self.decoder = Decoder(output_dim, d_model, nhead, num_layers, dim_feedforward)

    def forward(self, input, target):
        """
        - input: (B, W, input_dim)
        - target: (B, W, output_dim)
        - memory: (B, W, d_model)
        - out: (B, W, output_dim)
        """
        memory = self.encoder(input)
        out = self.decoder(target, memory)
        return out


# ┌──────────────────────────────────────────────────────────┐
#  Trainer
# └──────────────────────────────────────────────────────────┘
class Trainer:
    def __init__(self, model, optimizer, scheduler, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def step(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)
        output = self.model(input, target)
        return output

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0
        for input, target, label in dataloader:
            self.optimizer.zero_grad()
            pred = self.step(input, target)
            loss = F.mse_loss(pred, label.to(self.device))
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        return epoch_loss

    def evaluate(self, dataloader):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for input, target, label in dataloader:
                pred = self.step(input, target)
                loss = F.mse_loss(pred, label.to(self.device))
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


# ┌──────────────────────────────────────────────────────────┐
#  Main function
# └──────────────────────────────────────────────────────────┘
def main():
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

    # Hyperparameters
    batch_size = survey.routines.numeric(
        "Batch size",
        decimal=False,
    )
    num_epochs = survey.routines.numeric(
        "Number of epochs",
        decimal=False,
    )
    lr = survey.routines.numeric(
        "Learning rate (with scheduler; for Adam/AdamW)",
    )
    infimum_lr = survey.routines.numeric(
        "Infimum learning rate",
    )
    hparams = {
        "input_dim": 6,
        "output_dim": 1,
        "d_model": 0,
        "nhead": 0,
        "num_layers": 0,
        "dim_feedforward": 0,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "infimum_lr": infimum_lr
    }
    # ──────────────────────────────────────────────────────────────────────
    # Data
    train_data = load_ett_data("train")
    val_data   = load_ett_data("test")
    dl_train   = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dl_val     = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    # ──────────────────────────────────────────────────────────────────────
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.85, 0.98))
    scheduler = ExpHyperbolicLR(optimizer, upper_bound=250, max_iter=num_epochs, init_lr=lr, infimum_lr=infimum_lr)
    # ──────────────────────────────────────────────────────────────────────
    # Hyperparameter candidates
    candidates = {
        "d_model": [32, 64, 128],
        "nhead": [2, 4, 8],
        "dim_feedforward": [128, 256, 512],
        "num_layers": [2, 3, 4],
    }
    keys = list(candidates.keys())
    vals = list(candidates.values())
    product_vals = list(product(*vals))

    for vals in product_vals:
        hparams.update(dict(zip(keys, vals)))
        
        # Seeds from random.org
        seed = 42

        # Training with LR scheduler
        # Fix seed for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

        group_name = f"EH_{d_model}_{nhead}_{dim_feedforward}_{numlayers}({num_epochs})"
        run_name = f"{group_name}[{seed}]"
        tags = ["ExpHyperbolicLR", f"{num_epochs}"]

        model = TFEncDec(hparams)
        net = model.to(device)

        wandb.init(project="HyperbolicLR-ETT(Explore)", name=run_name, group=group_name, config=hparams, tags=tags)
        progress = Progress()

        trainer = Trainer(net, optimizer, scheduler, device=device)
        trainer.train(dl_train, dl_val, progress, num_epochs)
        wandb.finish()


if __name__ == "__main__":
    main()
