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

import wandb
import survey
import argparse
import itertools


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
    parser.add_argument('--project', type=str, default="HyperbolicLR-OSC")
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
    run_modes = ['Run', 'Search', 'Compare', 'Optimize', 'Optimize-Compare']
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

    # Survey dtype, mode
    dtypes = ["simple", "damped"]
    #modes  = ["S", "MS", "M"]
    dtype = survey.routines.select(
        "Select dtype",
        options=dtypes
    )
    dtype = dtypes[dtype]
    #mode = survey.routines.select(
    #    "Select mode",
    #    options=modes
    #)
    #mode = modes[mode]
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
        'scheduler': ExpHyperbolicLR,
        'scheduler_name': 'EH',
        'scheduler_params': {
            'upper_bound': 250,
            'max_iter': epochs,
            'init_lr': lr,
            'infimum_lr': infimum_lr
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

    if run_mode == 'Run':
        run(run_config, hparams, dl_train=dl_train, dl_val=dl_val, seeds=seeds, device=device)
    elif run_mode == 'Search':
        keys = list(hparams.keys())
        selections = survey.routines.basket(
            "Select hyperparameters to search (Use '->' to select)",
            options=keys
        )
        selections = [keys[select] for select in selections]

        search_params = {}
        for key in selections:
            val_cands = survey.routines.input(
                f"Input all candidates for {key} (Use ',' to separate)",
            )
            val_cands = [int(val) for val in val_cands.split(',')]
            search_params[key] = val_cands

        search_keys = list(search_params.keys())
        search_vals = list(search_params.values())

        for pval in itertools.product(*search_vals):
            hparams.update(dict(zip(search_keys, pval)))
            run(run_config, hparams, seeds=seeds, dl_train=dl_train, dl_val=dl_val, device=device)
    elif run_mode == 'Compare':
        optimizers = {
            #"SGD": optim.SGD,
            "Adam": optim.Adam,
            "AdamW": optim.AdamW,
        }
        optimizer_params_no_scheduler = {
            #"SGD": {"lr": lr / 10, "momentum": 0.9, "weight_decay": 5e-4},
            "Adam": {"lr": lr / 10},
            "AdamW": {"lr": lr / 10},
        }
        optimizer_params = {
            #"SGD": {"lr": lr, "momentum": 0.9, "weight_decay": 5e-4},
            "Adam": {"lr": lr},
            "AdamW": {"lr": lr},
        }
        schedulers = {
            "P": optim.lr_scheduler.PolynomialLR,
            "C": optim.lr_scheduler.CosineAnnealingLR,
            "E": optim.lr_scheduler.ExponentialLR,
            "H": HyperbolicLR,
            "EH": ExpHyperbolicLR,
        }
        scheduler_params = {
            "P": {"power": 0.5, "total_iters": epochs},
            "C": {"T_max": epochs, "eta_min": infimum_lr},
            "E": {"gamma": 0.96},
            "H": {"upper_bound": 250, "max_iter": epochs, "init_lr": lr, "infimum_lr": infimum_lr},
            "EH": {"upper_bound": 250, "max_iter": epochs, "init_lr": lr, "infimum_lr": infimum_lr},
        }
        def adjust_params_for_SGD(scheduler_name, params, optimizer_name):
            if optimizer_name == "SGD":
                adjusted_params = params.copy()
                if scheduler_name == "C":
                    adjusted_params["eta_min"] = params["eta_min"]
                elif scheduler_name in ["H", "EH"]:
                    adjusted_params["init_lr"] = params["init_lr"]
                    adjusted_params["infimum_lr"] = params["infimum_lr"]
                return adjusted_params
            return params

        for optimizer_name, optimizer_class in optimizers.items():
            # Training without LR scheduler
            run_config["optimizer"] = optimizer_class
            run_config["optimizer_params"] = optimizer_params_no_scheduler[optimizer_name]
            run_config["optimizer_name"] = optimizer_name
            run_config["scheduler"] = None
            run_config["scheduler_name"] = "N"
            run_config["scheduler_params"] = None
            run(run_config, hparams, seeds=seeds, dl_train=dl_train, dl_val=dl_val, device=device)

            # Training with LR scheduler
            for scheduler_name, scheduler_class in schedulers.items():
                run_config["optimizer_params"] = optimizer_params[optimizer_name]
                run_config["scheduler"] = scheduler_class
                run_config["scheduler_name"] = scheduler_name
                run_config["scheduler_params"] = adjust_params_for_SGD(scheduler_name, scheduler_params[scheduler_name], optimizer_name)
                run(run_config, hparams, seeds=seeds, dl_train=dl_train, dl_val=dl_val, device=device)
    elif run_mode == 'Optimize':
        optimizers = ["Adam", "AdamW"]
        choose_optimizer = survey.routines.select(
            "Select optimizer",
            options=optimizers
        )
        optimizer_name = optimizers[choose_optimizer]

        schedulers = ["N", "P", "C", "E", "H", "EH"]
        choose_scheduler = survey.routines.select(
            "Select scheduler",
            options=schedulers
        )
        scheduler_name = schedulers[choose_scheduler]

        schedulers = {
            "N": None,
            "P": optim.lr_scheduler.PolynomialLR,
            "C": optim.lr_scheduler.CosineAnnealingLR,
            "E": optim.lr_scheduler.ExponentialLR,
            "H": HyperbolicLR,
            "EH": ExpHyperbolicLR,
        }

        def object(trial):
            lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)

            run_config["optimizer"] = optim.Adam if optimizer_name == "Adam" else optim.AdamW
            run_config["optimizer_params"] = {"lr": lr}
            run_config["optimizer_name"] = optimizer_name
            run_config["scheduler"] = schedulers[scheduler_name]
            run_config["scheduler_name"] = scheduler_name

            hparams["lr"] = lr

            if scheduler_name == "N":
                run_config["scheduler_params"] = None
            elif scheduler_name == "P":
                power = trial.suggest_float("power", 0.5, 3.0)
                run_config["scheduler_params"] = {"power": power, "total_iters": epochs}
                hparams["power"] = power
            elif scheduler_name == "E":
                gamma = trial.suggest_float("gamma", 0.9, 0.99)
                run_config["scheduler_params"] = {"gamma": gamma}
                hparams["gamma"] = gamma
            elif scheduler_name == "C" or scheduler_name == "H" or scheduler_name == "EH":
                infimum_lr = trial.suggest_float("infimum_lr", 1e-7, 1e-4, log=True)
                hparams["infimum_lr"] = infimum_lr
                if scheduler_name == "C":
                    run_config["scheduler_params"] = {"T_max": epochs, "eta_min": infimum_lr}
                else:
                    upper_bound = trial.suggest_int("upper_bound", 200, 400, step=50)
                    hparams["upper_bound"] = upper_bound
                    run_config["scheduler_params"] = {"upper_bound": upper_bound, "max_iter": epochs, "init_lr": lr, "infimum_lr": infimum_lr}

            val_loss = run(run_config, hparams, seeds=seeds, dl_train=dl_train, dl_val=dl_val, device=device)
            return val_loss

        study_name = f"{project_name}-{optimizer_name}-{scheduler_name}"
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=study_name,
            storage=f"sqlite:///{project_name}.db",
            load_if_exists=True,
        )
        study.optimize(object, n_trials=25)
    elif run_mode == 'Optimize-Compare':
        # Load study
        studies = optuna.get_all_study_names(storage=f"sqlite:///{project_name}.db")

        # Select study
        choose_study = survey.routines.select(
            "Select study",
            options=studies
        )
        study_name = studies[choose_study]

        # Load study
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{project_name}.db")

        # Update project_name
        project_name = f"{project_name}[Compare]"
        run_config["project"] = project_name

        # Extract Optimizer & Scheduler
        pure_study_name = study_name.split(".")[0]
        scheduler_name = pure_study_name.split("-")[-1]
        optimizer_name = pure_study_name.split("-")[-2]

        schedulers = {
            "N": None,
            "P": optim.lr_scheduler.PolynomialLR,
            "C": optim.lr_scheduler.CosineAnnealingLR,
            "E": optim.lr_scheduler.ExponentialLR,
            "H": HyperbolicLR,
            "EH": ExpHyperbolicLR,
        }

        run_config["optimizer"] = optim.Adam if optimizer_name == "Adam" else optim.AdamW
        run_config["optimizer_name"] = optimizer_name
        run_config["scheduler"] = schedulers[scheduler_name]
        run_config["scheduler_name"] = scheduler_name

        # Best trial
        trial = study.best_trial
        print(trial)
        params = trial.params

        lr = params["lr"]
        hparams["lr"] = lr
        run_config["optimizer_params"] = {"lr": lr}

        # Run
        for epochs in [200, 150, 100, 50]:
            if scheduler_name == "N":
                run_config["scheduler_params"] = None
            elif scheduler_name == "P":
                power = params["power"]
                run_config["scheduler_params"] = {"power": power, "total_iters": epochs}
                hparams["power"] = power
            elif scheduler_name == "E":
                gamma = params["gamma"]
                run_config["scheduler_params"] = {"gamma": gamma}
                hparams["gamma"] = gamma
            elif scheduler_name == "C" or scheduler_name == "H" or scheduler_name == "EH":
                infimum_lr = params["infimum_lr"]
                hparams["infimum_lr"] = infimum_lr
                if scheduler_name == "C":
                    run_config["scheduler_params"] = {"T_max": epochs, "eta_min": infimum_lr}
                else:
                    upper_bound = params["upper_bound"]
                    run_config["scheduler_params"] = {"upper_bound": upper_bound, "max_iter": epochs, "init_lr": lr, "infimum_lr": infimum_lr}
                    hparams["upper_bound"] = upper_bound

            run_config["epochs"] = epochs
            run(run_config, hparams, seeds=seeds, dl_train=dl_train, dl_val=dl_val, device=device)
    else:
        raise ValueError("Can't see this error")


if __name__ == "__main__":
    main()
