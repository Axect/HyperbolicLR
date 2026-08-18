"""
Phase 2 decision-run worker: train one (task, scheduler, seed, budget, hparams)
configuration and write the full learning curve to a parquet file. No W&B dependency
(wandb runs in disabled mode); curves come from the Trainer history added in util.py.

Usage:
  python phase2_runner.py --task cifar10 --scheduler EH --seed 42 --epochs 100 \
      --lr 0.01 --infimum-lr 1e-6 --subset-ratio 0.2 --out results_phase2
  python phase2_runner.py --task integral --model DeepONet --scheduler EH --seed 42 \
      --epochs 200 --lr 0.005 --infimum-lr 1e-6 --out results_extrap

Scheduler code "SF" runs the LR-free AdamWScheduleFree optimizer (no scheduler).

Output: {out}/{stem}_{scheduler}_lr{lr}_inf{infimum}_N{epochs}_s{seed}[_B{tune}].parquet
where {stem} is the task name for cifar10/cifar100/osc (unchanged from the 2026-06
runs, so their parquets keep being skipped) and "integral-{model}" for the operator
task, whose model is a free axis. The optional "_B{tune}" suffix records that the
hyperparameters were tuned at a *different* budget than the one being trained here
(budget extrapolation), which also keeps two transfers of the same scheduler from
colliding on one filename when they only differ in a scheduler-specific parameter.

Columns: train_loss, val_loss, epoch, lr, [acc], task, model, scheduler, seed,
budget, lr0, infimum_lr, tune_budget, batch_size, sched_extra. `acc` exists for
classification only. `model`/`tune_budget`/`batch_size`/`sched_extra` were added
2026-08-17, so a directory that mixes older and newer parquets has a non-uniform
schema: read such a directory file-by-file (every script in scripts/ already does)
or with pl.concat(..., how="diagonal").

--measure-sharpness K (integral only) adds five more columns: lambda_max (largest Hessian
eigenvalue of one frozen training batch, NaN on unmeasured epochs), sharpness_iters,
sharpness_converged, sharpness_secs, and lr_pre_step. lr_pre_step is the step size in force
during that epoch, which is the one to pair with lambda_max; the `lr` column is read after
scheduler.step() and so holds the next epoch's value. --sharpness-precond adds
lambda_max_precond. None of these columns appear when the flag is off, so the schema of an
unmeasured run is unchanged, and the filename does not encode the flag either: measurement is
trajectory-neutral, so a measured and an unmeasured run of the same configuration hold the same
curve and belong at the same path.
"""
import argparse
import json
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import polars as pl
import wandb

from util import (load_cifar10, load_cifar100, load_osc_data, load_integral,
                  set_seed, Trainer, OperatorTrainer)
from model import SimpleCNN, LSTM_Seq2Seq, DeepONet, DeepONetLN, TFONet
from scheduler_config import create_scheduler, is_step_per_batch


# ── Task constants ───────────────────────────────────────────────────────────

TASKS = ["cifar10", "cifar100", "osc", "integral"]

# Fixed model per task, for the `model` parquet column. The operator task is the
# only one where the model is a command-line axis.
TASK_MODEL = {"cifar10": "SimpleCNN", "cifar100": "SimpleCNN", "osc": "LSTM_Seq2Seq"}

# Integral operator-learning task. Model hyperparameters are taken verbatim from
# main_integral.py, which does net = model(hparams, device=device) with hparams =
# model.default_hparams, i.e. DeepONet hidden_size 1024 / num_layers 6 /
# num_branch 50, TFONet d_model 128 / nhead 2 / dim_feedforward 512 /
# num_layers 3 / dropout 0.0. Optimizer AdamW (torch defaults betas
# (0.9, 0.999), eps 1e-8, weight_decay 0.01, which is what arXiv v1 §5.3 reports),
# trainer OperatorTrainer, loss F.mse_loss. Nothing is re-tuned here.
INTEGRAL_MODELS = {"DeepONet": DeepONet, "DeepONetLN": DeepONetLN, "TFONet": TFONet}
INTEGRAL_DATA_ROOT = "data"

# Per-task batch size. cifar/osc keep 128, the value the 2026-06 Phase 2 parquets
# were produced with. integral uses 100, which is what arXiv v1 reports for the
# operator task ("Batch sizes were set to 256 for CIFAR-10 and oscillation
# prediction tasks, and 100 for the operator learning task").
#
# The 128 in run_all.py TASK_DEFAULTS["integral"] carries no weight: that path feeds
# the bare 3-tensor list from load_integral straight into DataLoader, which makes a
# dataset of length 3, so len(dl) == 1 and batch_size never takes effect. That path
# trains full-batch at one optimizer step per epoch. It cannot be the regime behind
# the v1 numbers, since v1 states a batch size for this task at all, so build_data
# zips the tensors into a TensorDataset instead of copying the defect.
DEFAULT_BATCH_SIZE = {"cifar10": 128, "cifar100": 128, "osc": 128, "integral": 100}


# ── Task construction ────────────────────────────────────────────────────────

def build_data(task, subset_ratio=0.2, batch_size=None, hist=50, pred=50,
               data_root=INTEGRAL_DATA_ROOT):
    """Build the train/val dataloaders for `task`.

    Kept separate from build_model so an in-process HPO loop can load the data once and rebuild only the model per trial. ``batch_size=None`` takes the task's default.
    """
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[task]
    num_workers = 2
    if task in ("cifar10", "cifar100"):
        loader = load_cifar10 if task == "cifar10" else load_cifar100
        ds_train, ds_val = loader(subset_ratio)
    elif task == "osc":
        ds_train, ds_val = load_osc_data(dtype="simple", mode="S", hist=hist, pred=pred)
    elif task == "integral":
        train_tensors, val_tensors = load_integral(root=data_root)
        # load_integral returns a bare list [u, y, G(u)(y)] of (n, 100) tensors.
        # OperatorTrainer iterates `for u, y, Guy in dataloader`, which only yields
        # per-sample triples if the tensors are zipped into a TensorDataset first.
        ds_train = TensorDataset(*train_tensors)
        ds_val = TensorDataset(*val_tensors)
        # Already resident tensors: worker processes would only add IPC cost.
        num_workers = 0
    else:
        raise ValueError(task)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val


def task_acc(task):
    """Whether the task reports accuracy next to val loss (classification only)."""
    return task in ("cifar10", "cifar100")


def build_model(task, device, model_name=None, pred=50):
    """Build the model for `task`. Returns (model, criterion, acc, model_name)."""
    if task in ("cifar10", "cifar100"):
        num_classes = 10 if task == "cifar10" else 100
        model = SimpleCNN(SimpleCNN.default_hparams, num_classes=num_classes)
        return model, F.cross_entropy, task_acc(task), TASK_MODEL[task]
    if task == "osc":
        model = LSTM_Seq2Seq(LSTM_Seq2Seq.default_hparams, pred=pred, device=device)
        return model, F.mse_loss, False, TASK_MODEL[task]
    if task == "integral":
        name = model_name or "DeepONet"
        if name not in INTEGRAL_MODELS:
            raise ValueError(f"unknown integral model: {name}")
        cls = INTEGRAL_MODELS[name]
        model = cls(cls.default_hparams, device=device)
        # OperatorTrainer hardcodes F.mse_loss; returned for the record only.
        return model, F.mse_loss, False, name
    raise ValueError(task)


def build_task(task, subset_ratio, batch_size, device, hist=50, pred=50,
               model_name=None, data_root=INTEGRAL_DATA_ROOT):
    """batch_size=None takes DEFAULT_BATCH_SIZE[task]."""
    dl_train, dl_val = build_data(task, subset_ratio, batch_size, hist, pred, data_root)
    model, criterion, acc, model_name = build_model(task, device, model_name, pred)
    return model, dl_train, dl_val, criterion, acc, model_name


def build_optim(scheduler_code, model, lr, infimum_lr, epochs, extra,
                steps_per_epoch=1):
    if scheduler_code == "SF":
        from schedulefree import AdamWScheduleFree
        opt = AdamWScheduleFree(model.parameters(), lr=lr,
                                warmup_steps=extra.get("warmup_steps", 100))
        return opt, None, False
    opt = optim.AdamW(model.parameters(), lr=lr)
    # steps_per_epoch only reaches OneCycleLR/CyclicLR (see get_scheduler_params);
    # every other code ignores it, so passing the true value is a no-op for them.
    sched = create_scheduler(scheduler_code, opt, lr, infimum_lr, epochs,
                             steps_per_epoch=steps_per_epoch, **extra)
    return opt, sched, is_step_per_batch(scheduler_code)


def make_trainer(task, model, opt, sched, criterion, acc, device, step_per_batch,
                 measure_sharpness=0, sharpness_iters=20, sharpness_tol=1e-3,
                 sharpness_precond=False):
    """Sharpness measurement is wired for the operator task only.

    `Trainer` (cifar/osc) is deliberately left untouched: its models carry BatchNorm, whose
    running statistics an extra forward pass would move, so a measurement there could not be
    made trajectory-neutral as cheaply as it is for `OperatorTrainer`.
    """
    if task == "integral":
        return OperatorTrainer(model, opt, sched, device=device,
                               step_per_batch=step_per_batch,
                               measure_sharpness=measure_sharpness,
                               sharpness_iters=sharpness_iters,
                               sharpness_tol=sharpness_tol,
                               sharpness_precond=sharpness_precond)
    if measure_sharpness:
        raise ValueError(f"--measure-sharpness is only implemented for task=integral, got {task}")
    return Trainer(model, opt, sched, criterion=criterion, acc=acc,
                   device=device, step_per_batch=step_per_batch)


# ── Naming and scheduler extras ──────────────────────────────────────────────

def curve_tag(task, model_name, scheduler, lr, infimum_lr, epochs, seed,
              tune_budget=0):
    return curve_tag_parts(task, model_name, scheduler, f"{lr:g}", f"{infimum_lr:g}",
                           epochs, seed, tune_budget)


def curve_tag_parts(task, model_name, scheduler, lr_text, infimum_text, epochs, seed,
                    tune_budget=0):
    """curve_tag with lr / infimum_lr already rendered, for previewing a job list."""
    stem = f"{task}-{model_name}" if task == "integral" else task
    tag = (f"{stem}_{scheduler}_lr{lr_text}_inf{infimum_text}"
           f"_N{epochs}_s{seed}")
    if tune_budget:
        tag += f"_B{tune_budget}"
    return tag


def _coerce(value):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_sched_extra(items):
    """Parse repeated --sched-extra KEY=VALUE into a kwargs dict for create_scheduler.

    Needed for the scheduler parameters that Optuna tunes but that have no dedicated flag: P power, E gamma, L end_factor, S step_size/gamma, OC pct_start, CY step_size_up.
    """
    extra = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--sched-extra expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        extra[key.strip()] = _coerce(value.strip())
    return extra


def scheduler_extra(scheduler, upper_bound, cooldown_frac, sched_extra_items):
    extra = {}
    if scheduler in ("H", "EH", "WH", "WEH"):
        extra["upper_bound"] = upper_bound
    if scheduler == "WSD":
        extra["cooldown_frac"] = cooldown_frac
    extra.update(parse_sched_extra(sched_extra_items))
    return extra


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--model", default=None,
                   help="integral only: DeepONet | TFONet (default DeepONet). "
                        "Ignored for cifar10/cifar100/osc, whose model is fixed.")
    p.add_argument("--scheduler", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--infimum-lr", type=float, default=1e-6)
    p.add_argument("--subset-ratio", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=None,
                   help=f"default per task: {DEFAULT_BATCH_SIZE}")
    p.add_argument("--upper-bound", type=int, default=250)
    p.add_argument("--cooldown-frac", type=float, default=0.2)
    p.add_argument("--sched-extra", action="append", default=None,
                   metavar="KEY=VALUE",
                   help="extra create_scheduler kwarg, repeatable "
                        "(e.g. --sched-extra power=1.7 --sched-extra gamma=0.95)")
    p.add_argument("--tune-budget", type=int, default=0,
                   help="budget the hyperparameters were tuned at, when it differs "
                        "from --epochs. 0 means same-budget. Nonzero appends _B{n} "
                        "to the filename and is recorded in the tune_budget column.")
    p.add_argument("--data-root", default=INTEGRAL_DATA_ROOT,
                   help="integral only: directory holding train.parquet / val.parquet")
    p.add_argument("--measure-sharpness", type=int, default=0, metavar="K",
                   help="integral only: every K epochs, measure the largest Hessian eigenvalue "
                        "of one frozen training batch and record it (0 = off, the default, "
                        "which leaves both the curve and the parquet schema unchanged)")
    p.add_argument("--sharpness-iters", type=int, default=20,
                   help="power-iteration cap per measurement (default 20)")
    p.add_argument("--sharpness-tol", type=float, default=1e-3,
                   help="relative power-iteration convergence tolerance (default 1e-3)")
    p.add_argument("--sharpness-precond", action="store_true",
                   help="also record the sharpness of the AdamW-preconditioned Hessian "
                        "P^{-1} H, whose Adam stability threshold is 38 / lr "
                        "(arXiv:2207.14484) rather than 2 / lr. Doubles the measurement cost.")
    p.add_argument("--out", default="results_phase2")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | cuda:0 | cuda:1")
    args = p.parse_args()

    model_name = args.model if args.task == "integral" else TASK_MODEL[args.task]
    if args.task == "integral" and model_name is None:
        model_name = "DeepONet"
    tag = curve_tag(args.task, model_name, args.scheduler, args.lr,
                    args.infimum_lr, args.epochs, args.seed, args.tune_budget)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, tag + ".parquet")
    if os.path.exists(out_path):
        print(f"skip (exists): {out_path}")
        return

    wandb.init(mode="disabled")
    set_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    batch_size = args.batch_size or DEFAULT_BATCH_SIZE[args.task]
    model, dl_train, dl_val, criterion, acc, model_name = build_task(
        args.task, args.subset_ratio, batch_size, device,
        model_name=model_name, data_root=args.data_root)

    extra = scheduler_extra(args.scheduler, args.upper_bound, args.cooldown_frac,
                            args.sched_extra)
    opt, sched, per_batch = build_optim(args.scheduler, model, args.lr,
                                        args.infimum_lr, args.epochs, extra,
                                        steps_per_epoch=len(dl_train))

    trainer = make_trainer(args.task, model, opt, sched, criterion, acc, device,
                           per_batch, measure_sharpness=args.measure_sharpness,
                           sharpness_iters=args.sharpness_iters,
                           sharpness_tol=args.sharpness_tol,
                           sharpness_precond=args.sharpness_precond)
    trainer.train(dl_train, dl_val, epochs=args.epochs)

    df = pl.DataFrame(trainer.history)
    for k, v in [("task", args.task), ("model", model_name),
                 ("scheduler", args.scheduler),
                 ("seed", args.seed), ("budget", args.epochs),
                 ("lr0", args.lr), ("infimum_lr", args.infimum_lr),
                 ("tune_budget", args.tune_budget),
                 ("batch_size", batch_size),
                 ("sched_extra", json.dumps(extra, sort_keys=True))]:
        df = df.with_columns(pl.lit(v).alias(k))
    df.write_parquet(out_path)
    print(f"wrote {out_path} ({len(df)} epochs, final val_loss "
          f"{trainer.history[-1]['val_loss']:.4e})")


if __name__ == "__main__":
    main()
