"""
Phase 2 decision-run worker: train one (task, scheduler, seed, budget, hparams)
configuration and write the full learning curve to a parquet file. No W&B dependency
(wandb runs in disabled mode); curves come from the Trainer history added in util.py.

Usage:
  python phase2_runner.py --task cifar10 --scheduler EH --seed 42 --epochs 100 \
      --lr 0.01 --infimum-lr 1e-6 --subset-ratio 0.2 --out results_phase2

Scheduler code "SF" runs the LR-free AdamWScheduleFree optimizer (no scheduler).
Output: {out}/{task}_{scheduler}_lr{lr}_inf{infimum}_N{epochs}_s{seed}.parquet
"""
import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import polars as pl
import wandb

from util import load_cifar10, load_cifar100, load_osc_data, set_seed, Trainer
from model import SimpleCNN, LSTM_Seq2Seq
from scheduler_config import create_scheduler, is_step_per_batch


def build_task(task, subset_ratio, batch_size, device, hist=50, pred=50):
    if task in ("cifar10", "cifar100"):
        loader = load_cifar10 if task == "cifar10" else load_cifar100
        ds_train, ds_val = loader(subset_ratio)
        num_classes = 10 if task == "cifar10" else 100
        model = SimpleCNN(SimpleCNN.default_hparams, num_classes=num_classes)
        criterion = F.cross_entropy
        acc = True
    elif task == "osc":
        ds_train, ds_val = load_osc_data(dtype="simple", mode="S", hist=hist, pred=pred)
        model = LSTM_Seq2Seq(LSTM_Seq2Seq.default_hparams, pred=pred, device=device)
        criterion = F.mse_loss
        acc = False
    else:
        raise ValueError(task)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    return model, dl_train, dl_val, criterion, acc


def build_optim(scheduler_code, model, lr, infimum_lr, epochs, extra):
    if scheduler_code == "SF":
        from schedulefree import AdamWScheduleFree
        opt = AdamWScheduleFree(model.parameters(), lr=lr,
                                warmup_steps=extra.get("warmup_steps", 100))
        return opt, None, False
    opt = optim.AdamW(model.parameters(), lr=lr)
    sched = create_scheduler(scheduler_code, opt, lr, infimum_lr, epochs, **extra)
    return opt, sched, is_step_per_batch(scheduler_code)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["cifar10", "cifar100", "osc"])
    p.add_argument("--scheduler", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--infimum-lr", type=float, default=1e-6)
    p.add_argument("--subset-ratio", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--upper-bound", type=int, default=250)
    p.add_argument("--cooldown-frac", type=float, default=0.2)
    p.add_argument("--out", default="results_phase2")
    args = p.parse_args()

    tag = (f"{args.task}_{args.scheduler}_lr{args.lr:g}_inf{args.infimum_lr:g}"
           f"_N{args.epochs}_s{args.seed}")
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, tag + ".parquet")
    if os.path.exists(out_path):
        print(f"skip (exists): {out_path}")
        return

    wandb.init(mode="disabled")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, dl_train, dl_val, criterion, acc = build_task(
        args.task, args.subset_ratio, args.batch_size, device)

    extra = {}
    if args.scheduler in ("H", "EH", "WH", "WEH"):
        extra["upper_bound"] = args.upper_bound
    if args.scheduler == "WSD":
        extra["cooldown_frac"] = args.cooldown_frac
    opt, sched, per_batch = build_optim(args.scheduler, model, args.lr,
                                        args.infimum_lr, args.epochs, extra)

    trainer = Trainer(model, opt, sched, criterion=criterion, acc=acc,
                      device=device, step_per_batch=per_batch)
    trainer.train(dl_train, dl_val, epochs=args.epochs)

    df = pl.DataFrame(trainer.history)
    for k, v in [("task", args.task), ("scheduler", args.scheduler),
                 ("seed", args.seed), ("budget", args.epochs),
                 ("lr0", args.lr), ("infimum_lr", args.infimum_lr)]:
        df = df.with_columns(pl.lit(v).alias(k))
    df.write_parquet(out_path)
    print(f"wrote {out_path} ({len(df)} epochs, final val_loss "
          f"{trainer.history[-1]['val_loss']:.4e})")


if __name__ == "__main__":
    main()
