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

import argparse
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
    if mode == "train":
        df = pd.read_parquet("./data/ETTh1_train.parquet")
    elif mode == "test":
        df = pd.read_parquet("./data/ETTh1_test.parquet")
    else:
        raise ValueError("mode must be either 'train' or 'test'")

    features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    
    # Group the data
    grouped_data = df.groupby('group')[features]
    
    # Convert to list of tensors
    data_tensors = [torch.tensor(group.values, dtype=torch.float32) 
                    for _, group in grouped_data]
    
    # Stack tensors
    data = torch.stack(data_tensors)  # (N, seq_len, num_features)

    print(f"Data shape: {data.shape}")
    
    return data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ┌──────────────────────────────────────────────────────────┐
#  Network Model
# └──────────────────────────────────────────────────────────┘
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class iTransformer(nn.Module):
    def __init__(self, configs):
        super(iTransformer, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        # Encoding
        self.enc_embedding = nn.Linear(self.seq_len, configs.d_model)
        self.pos_encoder = PositionalEncoding(configs.d_model)

        # Encoder
        self.encoder = nn.ModuleList([
            EncoderLayer(configs.d_model, configs.nhead, configs.d_ff, configs.dropout)
            for _ in range(configs.e_layers)
        ])
        
        # Prediction
        self.projection = nn.Linear(configs.d_model, self.pred_len)

        # Normalization
        self.rev_norm = RevIN(self.enc_in)

    def forward(self, x):
        # x: [B, L, D]
        x = self.rev_norm(x, 'norm')
        
        # Encoding
        x = x.permute(0, 2, 1)  # x: [B, D, L]
        x = self.enc_embedding(x)  # x: [B, D, M]

        # Encoder
        x = x.permute(1, 0, 2)  # x: [D, B, M]
        x = self.pos_encoder(x)
        for layer in self.encoder:
            x = layer(x)
        
        # Prediction
        x = x.permute(1, 0, 2)  # x: [B, D, M]
        x = self.projection(x).permute(0, 2, 1)  # x: [B, L, D]
        
        x = self.rev_norm(x, 'denorm')
        return x


# ┌──────────────────────────────────────────────────────────┐
#  Trainer
# └──────────────────────────────────────────────────────────┘
class Trainer:
    def __init__(self, model, optimizer, scheduler, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def step(self, input):
        input = input.to(self.device)
        src = input[:, :-self.model.pred_len, :]
        tgt = input[:, -self.model.pred_len:, :]
        output = self.model(src)
        return output, tgt

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            self.optimizer.zero_grad()
            pred, tgt = self.step(batch)
            loss = F.mse_loss(pred, tgt)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                pred, tgt = self.step(batch)
                loss = F.mse_loss(pred, tgt)
                total_loss += loss.item()
        return total_loss / len(dataloader)

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
    parser = argparse.ArgumentParser(description='iTransformer for ETTh1 dataset')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    set_seed(args.seed)

    # Device selection
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
    print(f"Using device: {device}")

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

    # Hyperparameter candidates
    candidates = {
        "d_model": [32, 64, 128],
        "nhead": [2, 4, 8],
        "d_ff": [128, 256, 512],
        "e_layers": [2, 3],
    }

    # Grid search
    keys = list(candidates.keys())
    values = list(candidates.values())

    for combination in product(*values):
        # Fixed hyperparameters
        configs = type('Configs', (), {
            "seq_len": 168,
            "pred_len": 24,
            "enc_in": 7,
            "dropout": 0.0,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lr": lr,
            "infimum_lr": infimum_lr
        })()

        for i in range(len(keys)):
            setattr(configs, keys[i], combination[i])

        # Data loading
        train_data = load_ett_data("train")
        val_data = load_ett_data("test")
        dl_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        dl_val = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        # Model initialization
        model = iTransformer(configs).to(device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
        scheduler = ExpHyperbolicLR(optimizer, upper_bound=250, max_iter=num_epochs, init_lr=lr, infimum_lr=infimum_lr)

        # WandB metadata
        group_name = f"iT_{configs.d_model}_{configs.nhead}_{configs.d_ff}_{configs.e_layers}({num_epochs})"
        run_name = f"{group_name}[{args.seed}]"
        tags = ["iTransformer", f"{num_epochs}_epochs"]

        # WandB initialization
        wandb.init(project="HyperbolicLR-ETT-iTransformer(Explore)", config=vars(configs), group=group_name, name=run_name, tags=tags)

        # Training
        progress = Progress()
        trainer = Trainer(model, optimizer, scheduler, device=device)
        trainer.train(dl_train, dl_val, progress, num_epochs)
        
        wandb.finish()

if __name__ == "__main__":
    main()
