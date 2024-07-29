import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from hyperbolic_lr import HyperbolicLR, ExpHyperbolicLR

from model import LSTM_Seq2Seq
from util import load_osc_data, Trainer, set_seed

import optuna

# Samplers - for parameter searching
from optuna.samplers import TPESampler

from scipy.interpolate import PchipInterpolator
import numpy as np

import wandb
import survey
import argparse
import itertools


class SplineLR:
    def __init__(self, optimizer, max_iter, init_lr, min_lr, index=1, plus=True):
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.init_lr = init_lr

        if index < 1 or index > 3:
            raise ValueError("index should be 1 or 2 or 3")
        
        theta = np.arctan2(max_iter, init_lr - min_lr)
        alpha = np.min(np.pi / 2.0 - theta) * 2.0
        l = np.sqrt((init_lr - min_lr) ** 2 + max_iter ** 2) / 4.0

        delta_x = l * np.tan(alpha) * np.sin(theta)
        delta_y = l * np.tan(alpha) * np.cos(theta)

        x = np.linspace(0, max_iter, 5)
        y = np.linspace(init_lr, min_lr, 5)

        sign = 1 if plus else -1

        x[index] += delta_x * sign
        y[index] += delta_y * sign

        self.pchip = PchipInterpolator(x, y)
        self.iter = 0

    def step(self):
        return self._update_learning_rate()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self._optimizer.param_groups]

    def _get_lr(self):
        x = self.iter
        return np.exp(self.pchip(x))

    def _update_learning_rate(self):
        self.iter += 1
        lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def run(run_config, hparams, seeds, dl_train, dl_val, device='cpu'):
    project_name = run_config['project']
    model = run_config['model']
    optimizer = run_config['optimizer']
    optimizer_name = run_config['optimizer_name']
    optimizer_params = run_config['optimizer_params']
    scheduler = run_config['scheduler']
    scheduler_name = run_config['scheduler_name']
    scheduler_params = run_config['scheduler_params']
    epochs = run_config['epochs']
    batch_size = run_config['batch_size']
    dtype = run_config['dtype']
    mode = run_config['mode']
    hist = run_config['hist']
    pred = run_config['pred']
    input_size = run_config['input_size']
    output_size = run_config['output_size']

    group_name = f"{optimizer_name}_{scheduler_name}_{dtype}-{mode}-{hist}-{pred}({epochs})["
    tags = [dtype, mode, f"{hist}-{pred}", f"{epochs}", optimizer_name, scheduler_name]

    for key, val in hparams.items():
        # if type of val is float then {val:.4e} else val
        if type(val) == float:
            group_name += f"{key[0]}{val:.4e}-"
        else:
            group_name += f"{key[0]}{val}-"
    group_name = group_name[:-1] + "]"

    val_loss = 0.0
    for seed in seeds:
        set_seed(seed)

        run_name = f"{group_name}[{seed}]"

        net = model(hparams, pred=pred, input_size=input_size, output_size=output_size, device=device)
        optimizer_ = optimizer(net.parameters(), **optimizer_params)
        if scheduler is not None:
            scheduler_ = scheduler(optimizer_, **scheduler_params)
        else:
            scheduler_ = None

        wandb.init(
            project=project_name,
            name=run_name,
            group=group_name,
            tags=tags,
            config=hparams
        )

        trainer = Trainer(net, optimizer_, scheduler_, criterion=nn.HuberLoss(), acc=False, device=device)
        val_loss_ = trainer.train(dl_train, dl_val, epochs)
        val_loss += val_loss_
        wandb.finish()

    return val_loss / len(seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default="SplineLR-OSC")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    wandb.require("core")

    project_name = args.project
    seeds = [args.seed] if args.seed != 0 else [89, 231, 928, 814, 269]

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

    # Run mode
    run_modes = ['Test']
    run_mode = survey.routines.select(
        "Select run mode",
        options=run_modes
    )
    run_mode = run_modes[run_mode]

    # Data selection
    data = ["OSC"]
    data_index = survey.routines.select(
        "Select dataset",
        options=data
    )

    # Model selection
    models = ["LSTM_Seq2Seq"]
    model = survey.routines.select(
        "Select model",
        options=models
    )
    model = globals()[models[model]]

    # Survey dtype
    dtypes = ["simple", "damped", "total"]
    dtype = survey.routines.select(
        "Select dtype",
        options=dtypes
    )
    dtype = dtypes[dtype]

    # Mode
    mode = "S"
    if mode == "S":
        input_size = 1
        output_size = 1
    elif mode == "MS":
        input_size = 3
        output_size = 1
    else:
        input_size = 3
        output_size = 3

    # Survey hist, pred
    hist = survey.routines.numeric(
        "Input history length",
        decimal=False
    )
    pred = survey.routines.numeric(
        "Input prediction length",
        decimal=False
    )

    # Survey batch_size, epochs
    batch_size = survey.routines.numeric(
        "Input batch size",
        decimal=False
    )
    epochs = survey.routines.numeric(
        "Input epochs",
        decimal=False
    )

    # Survey learning rate & infimum_lr
    lr = survey.routines.numeric(
        "Input learning rate",
        decimal=True
    )
    infimum_lr = survey.routines.numeric(
        "Input infimum learning rate",
        decimal=True
    )


    # Run configuration
    run_config = {
        'project': project_name,
        'model': model,
        'optimizer': optim.AdamW,
        'optimizer_name': "AdamW",
        'optimizer_params': {
            'lr': lr,
        },
        'scheduler': SplineLR,
        'scheduler_name': '1p',
        'scheduler_params': {
            'max_iter': epochs,
            'init_lr': lr,
            'min_lr': infimum_lr,
            'index': 1,
            'plus': True,
        },
        'batch_size': batch_size,
        'epochs': epochs,
        'dtype': dtype,
        'mode': mode,
        'hist': hist,
        'pred': pred,
        'input_size': input_size,
        'output_size': output_size
    }

    # Load data
    ds_train, ds_val = load_osc_data(dtype, mode, hist, pred)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    hparams = model.default_hparams

    # Survey model specific hparams
    keys = list(hparams.keys())
    for key in keys:
        val = survey.routines.numeric(
            f"Input {key}",
            decimal=False
        )
        hparams[key] = val

    if run_mode == 'Test':
        optimizers = {
            "AdamW": optim.AdamW,
        }
        optimizer_params = {
            "AdamW": {"lr": lr},
        }
        schedulers = {
            "1p": SplineLR,
            "1n": SplineLR,
            "2p": SplineLR,
            "2n": SplineLR,
            "3p": SplineLR,
            "3n": SplineLR,
        }
        scheduler_params = {
            "1p": {"max_iter": epochs, "init_lr": lr, "min_lr": infimum_lr, "index": 1, "plus": True},
            "1n": {"max_iter": epochs, "init_lr": lr, "min_lr": infimum_lr, "index": 1, "plus": False},
            "2p": {"max_iter": epochs, "init_lr": lr, "min_lr": infimum_lr, "index": 2, "plus": True},
            "2n": {"max_iter": epochs, "init_lr": lr, "min_lr": infimum_lr, "index": 2, "plus": False},
            "3p": {"max_iter": epochs, "init_lr": lr, "min_lr": infimum_lr, "index": 3, "plus": True},
            "3n": {"max_iter": epochs, "init_lr": lr, "min_lr": infimum_lr, "index": 3, "plus": False},
        }

        for optimizer_name, optimizer_class in optimizers.items():
            # Training with LR scheduler
            run_config["optimizer"] = optimizer_class
            for scheduler_name, scheduler_class in schedulers.items():
                run_config["optimizer_params"] = optimizer_params[optimizer_name]
                run_config["scheduler"] = scheduler_class
                run_config["scheduler_name"] = scheduler_name
                run_config["scheduler_params"] = scheduler_params[scheduler_name]
                run(run_config, hparams, seeds=seeds, dl_train=dl_train, dl_val=dl_val, device=device)
    else:
        raise ValueError("Can't see this error")


if __name__ == "__main__":
    main()
