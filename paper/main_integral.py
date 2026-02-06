import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import DeepONet, TFONet
from util import load_integral, set_seed, OperatorTrainer
from scheduler_config import (
    get_schedulers, get_scheduler_names, get_scheduler_params,
    get_optuna_search_space, get_optimize_compare_params,
    create_scheduler, is_step_per_batch,
    get_lr_free_optimizers,
)

import optuna
from optuna.samplers import TPESampler

import wandb
import survey
import argparse


def run(run_config, hparams, seeds, dl_train, dl_val, device='cpu'):
    project_name = run_config['project']
    model = run_config['model']
    optimizer = run_config['optimizer']
    optimizer_name = run_config['optimizer_name']
    optimizer_params = run_config['optimizer_params']
    scheduler_name = run_config['scheduler_name']
    epochs = run_config['epochs']
    lr = optimizer_params.get('lr', 1.0)
    infimum_lr = run_config.get('infimum_lr', 1e-6)
    steps_per_epoch = run_config.get('steps_per_epoch', 1)
    scheduler_extra = run_config.get('scheduler_extra', {})

    step_per_batch = is_step_per_batch(scheduler_name)

    group_name = f"{optimizer_name}_{scheduler_name}({epochs})["
    tags = [f"{epochs}", optimizer_name, scheduler_name]

    for key, val in hparams.items():
        if type(val) == float:
            group_name += f"{key[0]}{val:.4e}-"
        else:
            group_name += f"{key[0]}{val}-"
    group_name = group_name[:-1] + "]"

    val_loss = 0.0
    for seed in seeds:
        set_seed(seed)

        run_name = f"{group_name}[{seed}]"

        net = model(hparams, device=device)
        optimizer_ = optimizer(net.parameters(), **optimizer_params)
        scheduler_ = create_scheduler(
            scheduler_name, optimizer_, lr, infimum_lr, epochs,
            steps_per_epoch=steps_per_epoch, **scheduler_extra
        )

        wandb.init(
            project=project_name,
            name=run_name,
            group=group_name,
            tags=tags,
            config=hparams
        )

        trainer = OperatorTrainer(
            net, optimizer_, scheduler_, device=device,
            step_per_batch=step_per_batch
        )
        val_loss += trainer.train(dl_train, dl_val, epochs)
        wandb.finish()

    return val_loss / len(seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default="HyperbolicLR-DeepONet")
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
    run_modes = ['Run', 'Compare', 'Optimize', 'Optimize-Compare']
    run_mode = survey.routines.select(
        "Select run mode",
        options=run_modes
    )
    run_mode = run_modes[run_mode]

    # Model selection
    models = ["DeepONet", "TFONet"]
    model = survey.routines.select(
        "Select model",
        options=models
    )
    model = globals()[models[model]]

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

    # Load data
    ds_train, ds_val = load_integral()
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    steps_per_epoch = len(dl_train)

    # Run configuration
    run_config = {
        'project': project_name,
        'model': model,
        'optimizer': optim.AdamW,
        'optimizer_name': "AdamW",
        'optimizer_params': {'lr': lr},
        'scheduler_name': 'EH',
        'scheduler_extra': {},
        'batch_size': batch_size,
        'epochs': epochs,
        'infimum_lr': infimum_lr,
        'steps_per_epoch': steps_per_epoch,
    }

    hparams = model.default_hparams.copy()

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
    elif run_mode == 'Compare':
        optimizers = {
            "Adam": optim.Adam,
            "AdamW": optim.AdamW,
        }
        scheduler_names = get_scheduler_names()

        for optimizer_name, optimizer_class in optimizers.items():
            for sched_name in scheduler_names:
                if sched_name == "N":
                    run_config["optimizer_params"] = {"lr": lr / 10}
                else:
                    run_config["optimizer_params"] = {"lr": lr}
                run_config["optimizer"] = optimizer_class
                run_config["optimizer_name"] = optimizer_name
                run_config["scheduler_name"] = sched_name
                run_config["scheduler_extra"] = {}
                run_config["infimum_lr"] = infimum_lr
                run(run_config, hparams, seeds=seeds, dl_train=dl_train, dl_val=dl_val, device=device)

        # LR-free optimizers
        lr_free = get_lr_free_optimizers()
        for opt_name, opt_class in lr_free.items():
            run_config["optimizer"] = opt_class
            run_config["optimizer_name"] = opt_name
            run_config["optimizer_params"] = {"lr": 1.0}
            run_config["scheduler_name"] = "N"
            run_config["scheduler_extra"] = {}
            run(run_config, hparams, seeds=seeds, dl_train=dl_train, dl_val=dl_val, device=device)

    elif run_mode == 'Optimize':
        optimizers = ["Adam", "AdamW"]
        choose_optimizer = survey.routines.select(
            "Select optimizer",
            options=optimizers
        )
        optimizer_name = optimizers[choose_optimizer]

        scheduler_names = get_scheduler_names()
        choose_scheduler = survey.routines.select(
            "Select scheduler",
            options=scheduler_names
        )
        scheduler_name = scheduler_names[choose_scheduler]

        def objective(trial):
            trial_lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)

            run_config["optimizer"] = optim.Adam if optimizer_name == "Adam" else optim.AdamW
            run_config["optimizer_params"] = {"lr": trial_lr}
            run_config["optimizer_name"] = optimizer_name
            run_config["scheduler_name"] = scheduler_name

            hparams["lr"] = trial_lr

            search_hparams = get_optuna_search_space(
                trial, scheduler_name, trial_lr, epochs, steps_per_epoch
            )
            hparams.update(search_hparams)

            run_config["infimum_lr"] = search_hparams.get("infimum_lr", infimum_lr)
            run_config["scheduler_extra"] = {
                k: v for k, v in search_hparams.items() if k != "infimum_lr"
            }

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
        study.optimize(objective, n_trials=25)

    elif run_mode == 'Optimize-Compare':
        # Load all studies
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

        run_config["optimizer"] = optim.Adam if optimizer_name == "Adam" else optim.AdamW
        run_config["optimizer_name"] = optimizer_name
        run_config["scheduler_name"] = scheduler_name

        # Best trial
        trial = study.best_trial
        print(trial)
        params = trial.params

        best_lr = params["lr"]
        hparams["lr"] = best_lr
        run_config["optimizer_params"] = {"lr": best_lr}

        # Run across epoch budgets
        for ep in [200, 150, 100, 50]:
            trial_infimum_lr, extra = get_optimize_compare_params(
                params, scheduler_name, best_lr, ep, steps_per_epoch
            )
            run_config["epochs"] = ep
            run_config["infimum_lr"] = trial_infimum_lr if trial_infimum_lr is not None else infimum_lr
            run_config["scheduler_extra"] = extra
            hparams.update({k: v for k, v in params.items() if k != "lr"})
            run(run_config, hparams, seeds=seeds, dl_train=dl_train, dl_val=dl_val, device=device)
    else:
        raise ValueError("Can't see this error")


if __name__ == "__main__":
    main()
