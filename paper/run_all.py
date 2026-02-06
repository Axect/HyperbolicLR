"""
run_all.py — All-in-One experiment runner for HyperbolicLR paper.

Non-interactive script that automates the full experiment flow:
  Phase 1: Optimize all schedulers via Optuna TPE
  Phase 2: Epoch sensitivity comparison (Optimize-Compare)

Usage:
  uv run python run_all.py --task cifar10 --model ResNetCIFAR
  uv run python run_all.py --task osc
  uv run python run_all.py --task integral --model DeepONet
  uv run python run_all.py --task cifar10 --model ResNetCIFAR --n-trials 10 --epochs 100
"""

import argparse
import json
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import optuna
from optuna.samplers import TPESampler

import wandb

from scheduler_config import (
    get_scheduler_names, get_optuna_search_space, get_optimize_compare_params,
    get_lr_free_optimizers, get_lr_free_optuna_search_space,
)


# ── Task Defaults ───────────────────────────────────────────────────────────

TASK_DEFAULTS = {
    "cifar10": {
        "model": "ResNetCIFAR", "lr": 0.01, "batch_size": 128,
        "epochs": 50, "infimum_lr": 1e-6, "subset_ratio": 1.0,
        "dataset": "CIFAR10", "num_classes": 10,
    },
    "osc": {
        "model": "LSTM_Seq2Seq", "lr": 0.01, "batch_size": 64,
        "epochs": 50, "infimum_lr": 1e-6,
        "dtype": "simple", "mode": "S", "hist": 50, "pred": 50,
    },
    "integral": {
        "model": "DeepONet", "lr": 0.005, "batch_size": 128,
        "epochs": 50, "infimum_lr": 1e-6,
    },
}


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="HyperbolicLR All-in-One Experiment Runner"
    )
    p.add_argument("--task", type=str, default="cifar10",
                   choices=["cifar10", "osc", "integral"])
    p.add_argument("--model", type=str, default=None,
                   help="Model name (default: task-specific)")
    p.add_argument("--optimizer", type=str, default="AdamW",
                   choices=["Adam", "AdamW"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--infimum-lr", type=float, default=None)
    p.add_argument("--n-trials", type=int, default=25)
    p.add_argument("--n-seeds", type=int, default=1,
                   help="Seeds during optimization (1=fast, 5=robust)")
    p.add_argument("--compare-seeds", type=int, default=5,
                   help="Seeds during Optimize-Compare phase")
    p.add_argument("--epoch-budgets", type=str, default="50,100,150,200",
                   help="Comma-separated epoch budgets for comparison")
    p.add_argument("--project", type=str, default=None,
                   help="W&B project name (auto-generated if omitted)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--schedulers", type=str, default=None,
                   help="Comma-separated subset of schedulers to run")
    p.add_argument("--skip-lr-free", action="store_true",
                   help="Skip LR-free optimizers (Prodigy, DAdapt)")
    p.add_argument("--skip-compare", action="store_true",
                   help="Skip Phase 2 (only optimize)")

    # Task-specific overrides
    p.add_argument("--dataset", type=str, default=None,
                   choices=["CIFAR10", "CIFAR100"])
    p.add_argument("--subset-ratio", type=float, default=None)
    p.add_argument("--dtype", type=str, default=None,
                   choices=["simple", "damped", "total"])
    p.add_argument("--hist", type=int, default=None)
    p.add_argument("--pred", type=int, default=None)

    return p.parse_args()


# ── Device ──────────────────────────────────────────────────────────────────

def get_device(args):
    if args.device is not None:
        return args.device
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


# ── Data Loading ────────────────────────────────────────────────────────────

def load_task_data(args, cfg):
    """Load data and build run_config / hparams for the given task."""

    if args.task == "cifar10":
        from util import load_cifar10, load_cifar100
        from model import SimpleCNN, ResNetCIFAR, SimpleViT

        model_registry = {
            "SimpleCNN": SimpleCNN,
            "ResNetCIFAR": ResNetCIFAR,
            "SimpleViT": SimpleViT,
        }
        model_cls = model_registry[cfg["model"]]

        dataset = cfg.get("dataset", "CIFAR10")
        subset_ratio = cfg.get("subset_ratio", 1.0)
        if dataset == "CIFAR10":
            ds_train, ds_val = load_cifar10(subset_ratio=subset_ratio)
            num_classes = 10
        else:
            ds_train, ds_val = load_cifar100(subset_ratio=subset_ratio)
            num_classes = 100

        num_workers = 4 if torch.cuda.is_available() else 0
        dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
        dl_val = DataLoader(ds_val, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)

        hparams = model_cls.default_hparams.copy()
        run_config = {
            "model": model_cls,
            "batch_size": cfg["batch_size"],
            "epochs": cfg["epochs"],
            "num_classes": num_classes,
            "infimum_lr": cfg["infimum_lr"],
            "steps_per_epoch": len(dl_train),
        }

    elif args.task == "osc":
        from util import load_osc_data
        from model import LSTM_Seq2Seq

        model_cls = LSTM_Seq2Seq
        dtype = cfg.get("dtype", "simple")
        mode = cfg.get("mode", "S")
        hist = cfg.get("hist", 50)
        pred = cfg.get("pred", 50)

        if mode == "S":
            input_size, output_size = 1, 1
        elif mode == "MS":
            input_size, output_size = 3, 1
        else:
            input_size, output_size = 3, 3

        ds_train, ds_val = load_osc_data(dtype, mode, hist, pred)
        dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"],
                              shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=cfg["batch_size"],
                            shuffle=False)

        hparams = model_cls.default_hparams.copy()
        run_config = {
            "model": model_cls,
            "batch_size": cfg["batch_size"],
            "epochs": cfg["epochs"],
            "dtype": dtype,
            "mode": mode,
            "hist": hist,
            "pred": pred,
            "input_size": input_size,
            "output_size": output_size,
            "infimum_lr": cfg["infimum_lr"],
            "steps_per_epoch": len(dl_train),
        }

    elif args.task == "integral":
        from util import load_integral
        from model import DeepONet, TFONet

        model_registry = {"DeepONet": DeepONet, "TFONet": TFONet}
        model_cls = model_registry[cfg["model"]]

        ds_train, ds_val = load_integral()
        dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"],
                              shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=cfg["batch_size"],
                            shuffle=False)

        hparams = model_cls.default_hparams.copy()
        run_config = {
            "model": model_cls,
            "batch_size": cfg["batch_size"],
            "epochs": cfg["epochs"],
            "infimum_lr": cfg["infimum_lr"],
            "steps_per_epoch": len(dl_train),
        }

    else:
        raise ValueError(f"Unknown task: {args.task}")

    return dl_train, dl_val, run_config, hparams


# ── run_fn selection ────────────────────────────────────────────────────────

def get_run_fn(task):
    if task == "cifar10":
        from main_cifar10 import run
    elif task == "osc":
        from main_osc import run
    elif task == "integral":
        from main_integral import run
    else:
        raise ValueError(f"Unknown task: {task}")
    return run


def extract_val_loss(task, result):
    """CIFAR10 returns (val_loss, accuracy); others return scalar."""
    if task == "cifar10":
        return result[0]
    return result


# ── Phase 1: Optimize ──────────────────────────────────────────────────────

def optimize_one(args, run_fn, run_config, hparams, dl_train, dl_val,
                 optimizer_name, scheduler_name, device, cfg):
    """Optimize one scheduler with Optuna TPE."""
    project_name = cfg["project"]
    epochs = cfg["epochs"]
    infimum_lr = cfg["infimum_lr"]
    steps_per_epoch = run_config["steps_per_epoch"]
    opt_seeds = list(range(1, args.n_seeds + 1))

    def objective(trial):
        h = hparams.copy()
        trial_lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)

        rc = run_config.copy()
        rc["project"] = project_name
        rc["optimizer"] = optim.Adam if optimizer_name == "Adam" else optim.AdamW
        rc["optimizer_params"] = {"lr": trial_lr}
        rc["optimizer_name"] = optimizer_name
        rc["scheduler_name"] = scheduler_name
        rc["scheduler_extra"] = {}
        rc["epochs"] = epochs

        h["lr"] = trial_lr

        search_hparams = get_optuna_search_space(
            trial, scheduler_name, trial_lr, epochs, steps_per_epoch
        )
        h.update(search_hparams)

        rc["infimum_lr"] = search_hparams.get("infimum_lr", infimum_lr)
        rc["scheduler_extra"] = {
            k: v for k, v in search_hparams.items() if k != "infimum_lr"
        }

        result = run_fn(rc, h, seeds=opt_seeds, dl_train=dl_train,
                        dl_val=dl_val, device=device)
        return extract_val_loss(args.task, result)

    study_name = f"{project_name}-{optimizer_name}-{scheduler_name}"
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=study_name,
        storage=f"sqlite:///{project_name}.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)
    return study


def optimize_lr_free_one(args, run_fn, run_config, hparams, dl_train, dl_val,
                         opt_name, device, cfg):
    """Optimize one LR-free optimizer with Optuna TPE."""
    project_name = cfg["project"]
    epochs = cfg["epochs"]
    lr_free_opts = get_lr_free_optimizers()
    opt_class = lr_free_opts[opt_name]
    opt_seeds = list(range(1, args.n_seeds + 1))

    def objective(trial):
        h = hparams.copy()

        extra_params = get_lr_free_optuna_search_space(trial, opt_name)

        rc = run_config.copy()
        rc["project"] = project_name
        rc["optimizer"] = opt_class
        rc["optimizer_params"] = {"lr": 1.0, **extra_params}
        rc["optimizer_name"] = opt_name
        rc["scheduler_name"] = "N"
        rc["scheduler_extra"] = {}
        rc["epochs"] = epochs

        result = run_fn(rc, h, seeds=opt_seeds, dl_train=dl_train,
                        dl_val=dl_val, device=device)
        return extract_val_loss(args.task, result)

    study_name = f"{project_name}-{opt_name}-N"
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=study_name,
        storage=f"sqlite:///{project_name}.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)
    return study


# ── Phase 2: Optimize-Compare ──────────────────────────────────────────────

def compare_one(args, run_fn, run_config, hparams, dl_train, dl_val,
                study, scheduler_name, optimizer_name, device, cfg,
                epoch_budgets, is_lr_free=False):
    """Run best params from a study across epoch budgets."""
    project_name = f"{cfg['project']}[Compare]"
    infimum_lr = cfg["infimum_lr"]
    steps_per_epoch = run_config["steps_per_epoch"]
    compare_seeds = list(range(1, args.compare_seeds + 1))

    trial = study.best_trial
    params = trial.params

    results = {}
    for ep in epoch_budgets:
        h = hparams.copy()
        rc = run_config.copy()
        rc["project"] = project_name
        rc["epochs"] = ep

        if is_lr_free:
            lr_free_opts = get_lr_free_optimizers()
            rc["optimizer"] = lr_free_opts[optimizer_name]
            rc["optimizer_name"] = optimizer_name
            rc["optimizer_params"] = {"lr": 1.0}
            # Restore LR-free extra params
            for k, v in params.items():
                rc["optimizer_params"][k] = v
            rc["scheduler_name"] = "N"
            rc["scheduler_extra"] = {}
            rc["infimum_lr"] = infimum_lr
        else:
            best_lr = params["lr"]
            h["lr"] = best_lr
            rc["optimizer"] = optim.Adam if optimizer_name == "Adam" else optim.AdamW
            rc["optimizer_name"] = optimizer_name
            rc["optimizer_params"] = {"lr": best_lr}
            rc["scheduler_name"] = scheduler_name

            trial_infimum_lr, extra = get_optimize_compare_params(
                params, scheduler_name, best_lr, ep, steps_per_epoch
            )
            rc["infimum_lr"] = trial_infimum_lr if trial_infimum_lr is not None else infimum_lr
            rc["scheduler_extra"] = extra
            h.update({k: v for k, v in params.items() if k != "lr"})

        result = run_fn(rc, h, seeds=compare_seeds, dl_train=dl_train,
                        dl_val=dl_val, device=device)
        val_loss = extract_val_loss(args.task, result)
        results[str(ep)] = val_loss

    return results


# ── Summary ─────────────────────────────────────────────────────────────────

def print_summary(optimization_results, comparison_results, epoch_budgets):
    ep_headers = [f"Ep{ep}" for ep in epoch_budgets]
    header = f"{'Scheduler':<16}| " + " | ".join(f"{h:>7}" for h in ep_headers)
    if comparison_results:
        header += f" | {'ΔSLCD':>7}"
    sep = "─" * 16 + "┼" + "┼".join("─" * 9 for _ in ep_headers)
    if comparison_results:
        sep += "┼" + "─" * 9

    print()
    print("═" * 58)
    print("  Summary")
    print("═" * 58)

    # Phase 1 results
    print()
    print("Phase 1: Best Optimization Results")
    print("─" * 58)
    print(f"{'Scheduler':<16}| {'Best val_loss':>14} | {'Best LR':>10}")
    print("─" * 16 + "┼" + "─" * 16 + "┼" + "─" * 12)
    for name, info in optimization_results.items():
        lr_str = f"{info['best_params'].get('lr', 1.0):.4e}" if 'lr' in info['best_params'] else "self-tuned"
        print(f"{name:<16}| {info['best_val_loss']:>14.4f} | {lr_str:>10}")

    # Phase 2 results
    if comparison_results:
        print()
        print("Phase 2: Epoch Sensitivity Comparison")
        print("─" * 58)
        print(header)
        print(sep)
        for name, ep_results in comparison_results.items():
            vals = [ep_results.get(str(ep), float('nan')) for ep in epoch_budgets]
            row = f"{name:<16}| " + " | ".join(f"{v:>7.4f}" for v in vals)
            # SLCD: relative change from max to min epoch
            if len(vals) >= 2 and vals[0] != 0:
                slcd = abs(vals[-1] - vals[0]) / abs(vals[0]) * 100
                row += f" | {slcd:>6.1f}%"
            print(row)

    print()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Merge task defaults with CLI overrides
    defaults = TASK_DEFAULTS[args.task].copy()
    if args.model is not None:
        defaults["model"] = args.model
    if args.epochs is not None:
        defaults["epochs"] = args.epochs
    if args.batch_size is not None:
        defaults["batch_size"] = args.batch_size
    if args.lr is not None:
        defaults["lr"] = args.lr
    if args.infimum_lr is not None:
        defaults["infimum_lr"] = args.infimum_lr
    if args.dataset is not None:
        defaults["dataset"] = args.dataset
    if args.subset_ratio is not None:
        defaults["subset_ratio"] = args.subset_ratio
    if args.dtype is not None:
        defaults["dtype"] = args.dtype
    if args.hist is not None:
        defaults["hist"] = args.hist
    if args.pred is not None:
        defaults["pred"] = args.pred

    cfg = defaults

    # Project name
    if args.project is not None:
        cfg["project"] = args.project
    else:
        cfg["project"] = f"HyperbolicLR-{args.task.upper()}-{cfg['model']}"

    # Device
    device = get_device(args)

    # Epoch budgets
    epoch_budgets = [int(x) for x in args.epoch_budgets.split(",")]

    # W&B
    wandb.require("core")
    import os
    os.environ["WANDB_MODE"] = args.wandb_mode

    # Scheduler list
    scheduler_names = get_scheduler_names()
    if args.schedulers is not None:
        selected = [s.strip() for s in args.schedulers.split(",")]
        scheduler_names = [s for s in scheduler_names if s in selected]

    # LR-free optimizers
    lr_free_opts = get_lr_free_optimizers() if not args.skip_lr_free else {}

    # Print banner
    total = len(scheduler_names) + len(lr_free_opts)
    print()
    print("═" * 58)
    print("  HyperbolicLR All-in-One Experiment")
    print(f"  Task: {args.task} | Model: {cfg['model']} | Optimizer: {args.optimizer}")
    print(f"  Epochs: {cfg['epochs']} | Trials: {args.n_trials} | Device: {device}")
    print(f"  Schedulers: {len(scheduler_names)} | LR-free: {len(lr_free_opts)} | Total: {total}")
    print("═" * 58)

    # Load data
    print("\nLoading data...")
    run_fn = get_run_fn(args.task)
    dl_train, dl_val, run_config, hparams = load_task_data(args, cfg)

    # ── Phase 1: Optimize ───────────────────────────────────────────────────
    print()
    print("Phase 1: Optimization (Optuna TPE)")
    print("─" * 58)

    optimization_results = {}
    studies = {}
    idx = 0

    for sched_name in scheduler_names:
        idx += 1
        print(f"[{idx}/{total}] Optimizing {sched_name}...")
        study = optimize_one(
            args, run_fn, run_config, hparams.copy(), dl_train, dl_val,
            args.optimizer, sched_name, device, cfg
        )
        best = study.best_trial
        studies[sched_name] = (study, args.optimizer, False)
        optimization_results[sched_name] = {
            "best_val_loss": best.value,
            "best_params": best.params,
        }
        print(f"       Best: val_loss={best.value:.4f}, "
              + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in best.params.items()))

    for opt_name in lr_free_opts:
        idx += 1
        print(f"[{idx}/{total}] Optimizing {opt_name} (LR-free)...")
        study = optimize_lr_free_one(
            args, run_fn, run_config, hparams.copy(), dl_train, dl_val,
            opt_name, device, cfg
        )
        best = study.best_trial
        studies[opt_name] = (study, opt_name, True)
        optimization_results[opt_name] = {
            "best_val_loss": best.value,
            "best_params": best.params,
        }
        print(f"       Best: val_loss={best.value:.4f}, "
              + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in best.params.items()))

    # ── Phase 2: Optimize-Compare ───────────────────────────────────────────
    comparison_results = {}
    if not args.skip_compare:
        print()
        print("Phase 2: Epoch Sensitivity Comparison")
        print("─" * 58)

        idx = 0
        for name, (study, opt_name, is_lr_free) in studies.items():
            idx += 1
            sched_name = "N" if is_lr_free else name
            print(f"[{idx}/{total}] Comparing {name} across {epoch_budgets} epochs...")
            ep_results = compare_one(
                args, run_fn, run_config, hparams.copy(), dl_train, dl_val,
                study, sched_name, opt_name, device, cfg,
                epoch_budgets, is_lr_free=is_lr_free
            )
            comparison_results[name] = ep_results

    # ── Summary ─────────────────────────────────────────────────────────────
    print_summary(optimization_results, comparison_results, epoch_budgets)

    # ── Save Results ────────────────────────────────────────────────────────
    output = {
        "config": {
            k: v for k, v in cfg.items() if not callable(v)
        },
        "optimization": optimization_results,
        "comparison": comparison_results,
    }
    output_path = f"results_{cfg['project']}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
