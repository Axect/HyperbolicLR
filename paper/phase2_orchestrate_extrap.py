"""
Budget-extrapolation orchestrator: tune a schedule at a small budget, deploy it at a
larger one, and keep the whole learning curve of every deployment.

    for scheduler in SCHEDULERS:
        for B_tune in TUNE_BUDGETS:
            best = TPE HPO at budget B_tune, objective = final val_loss
            for N in DEPLOY_BUDGETS:
                for seed in SEEDS:
                    train(best, budget=N, seed=seed) -> one parquet

This is the measurement behind the v12 headline: rescaling schedules (C, P, L, WC,
WSD) hold a high LR for a time proportional to N, so a setting that was safe at
B_tune crosses the instability boundary at N >> B_tune, while the log-space anchored
ExpHyperbolicLR has a nearly budget-independent dwell time. The falsifiable
prediction under test is that WSD, whose dwell ratio T(200)/T(50) = 3.93, collapses
with the rescaling family rather than with EH.

HPO runs in this process (one data load, one model rebuild per trial). Deployment
runs through phase2_runner.py as a subprocess so its skip-if-exists rule makes the
whole campaign resumable at parquet granularity.

No pruner: every trial runs to B_tune. The object of study is how a tuning decision
transfers across budgets, so trials must not be killed by a rule whose own fidelity
is the thing in question. Note that optuna.create_study(pruner=None) installs a
MedianPruner, so NopPruner is passed explicitly.

Search space: scheduler_config.get_optuna_search_space, with lr on the v1 arXiv range
1e-4 to 5e-2 (log). For the record, that function's per-scheduler ranges are
P power 0.5-3.0, E gamma 0.9-0.99, C/H/EH/WC/WSD infimum_lr 1e-7 to 1e-4 (log),
H/EH upper_bound 200-400 step 50, WSD cooldown_frac 0.1-0.4, L end_factor 1e-3 to
1e-1 (log). SF (AdamWScheduleFree) is tuned on lr only, because phase2_runner.py
takes no warmup_steps flag and pins it at 100.

Sharding is by scheduler, so one scheduler's HPO and all of its deployments stay in
one shard. Schedulers are independent, and this also keeps two GPUs off the same
sqlite file.

Regime, fixed to match arXiv v1 rather than the paper/ port: integral trains in
minibatches of 100 (v1 §5.3 states 100 for the operator task) over the 8000-sample
train split, i.e. 80 optimizer steps per epoch, and the deployment seeds are v1's
five random.org seeds. phase2_runner.build_data zips load_integral's three tensors
into a TensorDataset to get there; feeding that bare list to DataLoader, as
main_integral.py and run_all.py do, silently trains full-batch at one step per epoch
and ignores batch_size entirely.

Run from the paper/ directory: phase2_runner.py, data/ and the default out and
study-db paths are all relative to the working directory, as in
phase2_orchestrate_e1.py.

Usage:
  python phase2_orchestrate_extrap.py --task integral --model DeepONet \
      --device cuda:0 --shard 0/2 --out results_extrap/integral-DeepONet
  python phase2_orchestrate_extrap.py --task integral --dry-run
"""
import argparse
import math
import os
import subprocess
import sys
import time

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState
import torch
import wandb

from phase2_runner import (DEFAULT_BATCH_SIZE, INTEGRAL_MODELS, TASK_MODEL, TASKS,
                           build_data, build_model, build_optim, curve_tag_parts,
                           make_trainer, task_acc)
from scheduler_config import get_optimize_compare_params, get_optuna_search_space
from util import set_seed


# ── Campaign defaults ────────────────────────────────────────────────────────

# The dwell-time table in PUBLICATION_PLAN.md covers N, WC, WSD, C, P, H, EH, E.
# L is added as a second rescaling data point (dwell proportional to N, like C/P).
SCHEDULERS = ["N", "E", "P", "C", "L", "H", "EH", "WC", "WSD"]
TUNE_BUDGETS = [10, 50]
DEPLOY_BUDGETS = [50, 200]
# The five random.org seeds of arXiv v1, so the new integral table can be read next
# to the v1 DeepONet mean/std improvement figures. Phase 2's cifar/osc seeds were
# 7/42/1024; pass --seeds to use those instead.
SEEDS = [89, 231, 928, 814, 269]
N_TRIALS = 25
LR_RANGE = (1e-4, 5e-2)
INF_DEFAULT = 1e-6
HPO_SEED = 42
SAMPLER_SEED = 42
# Objective value handed to TPE when a trial's final val_loss is nan/inf. A finite
# sentinel keeps the study alive; optuna rejects nan outright.
DIVERGED = 1e10


# ── Search space ─────────────────────────────────────────────────────────────

def search_space(trial, scheduler, epochs, steps_per_epoch):
    """Suggest lr plus the scheduler's own parameters. Returns (lr, infimum_lr, extra)."""
    lr = trial.suggest_float("lr", LR_RANGE[0], LR_RANGE[1], log=True)
    if scheduler == "SF":
        return lr, INF_DEFAULT, {}
    space = get_optuna_search_space(trial, scheduler, lr, epochs, steps_per_epoch)
    infimum_lr = space.get("infimum_lr", INF_DEFAULT)
    extra = {k: v for k, v in space.items() if k != "infimum_lr"}
    return lr, infimum_lr, extra


def deploy_params(scheduler, params, deploy_budget, steps_per_epoch):
    """Map a best trial's params onto (lr, infimum_lr, extra) at the deploy budget."""
    lr = params["lr"]
    if scheduler == "SF":
        return lr, INF_DEFAULT, {}
    infimum_lr, extra = get_optimize_compare_params(
        params, scheduler, lr, deploy_budget, steps_per_epoch
    )
    if infimum_lr is None:
        infimum_lr = INF_DEFAULT
    return lr, infimum_lr, extra


# ── Study storage ────────────────────────────────────────────────────────────

def study_path(study_db, scheduler):
    """One sqlite file per scheduler unless --study-db names a single .db file."""
    if study_db.endswith(".db"):
        return study_db
    return os.path.join(study_db, f"{scheduler}.db")


def study_name_of(task, model_name, scheduler, tune_budget):
    return f"{task}-{model_name}-{scheduler}-B{tune_budget}"


def completed_trials(path, name):
    if not os.path.exists(path):
        return 0
    try:
        study = optuna.load_study(study_name=name, storage=f"sqlite:///{path}")
    except Exception:
        return 0
    return sum(1 for t in study.trials if t.state == TrialState.COMPLETE)


# ── HPO ──────────────────────────────────────────────────────────────────────

def run_hpo(task, model_name, scheduler, tune_budget, n_trials, dl_train, dl_val,
            device, acc, hpo_seed, storage, study_name, pred=50):
    steps_per_epoch = len(dl_train)

    def objective(trial):
        lr, infimum_lr, extra = search_space(trial, scheduler, tune_budget,
                                             steps_per_epoch)
        set_seed(hpo_seed)
        model, criterion, _, _ = build_model(task, device, model_name, pred)
        opt, sched, per_batch = build_optim(scheduler, model, lr, infimum_lr,
                                            tune_budget, extra,
                                            steps_per_epoch=steps_per_epoch)
        trainer = make_trainer(task, model, opt, sched, criterion, acc, device,
                               per_batch)
        result = trainer.train(dl_train, dl_val, epochs=tune_budget)
        val_loss = result[0] if isinstance(result, tuple) else result
        if not math.isfinite(val_loss):
            print(f"    trial {trial.number}: non-finite val_loss, scored "
                  f"{DIVERGED:g}", flush=True)
            return DIVERGED
        return val_loss

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SAMPLER_SEED),
        pruner=optuna.pruners.NopPruner(),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    done = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    remaining = max(0, n_trials - done)
    if remaining == 0:
        print(f"  HPO {study_name}: {done}/{n_trials} trials already complete, "
              f"reusing", flush=True)
    else:
        print(f"  HPO {study_name}: {done} complete, running {remaining} more "
              f"at budget {tune_budget}", flush=True)
        study.optimize(objective, n_trials=remaining)
    return study


# ── Deployment ───────────────────────────────────────────────────────────────

def deploy_cmd(task, model_name, scheduler, lr, infimum_lr, extra, budget, seed,
               tune_budget, out, device, subset_ratio, batch_size, data_root):
    cmd = [sys.executable, "phase2_runner.py",
           "--task", task, "--scheduler", scheduler,
           "--lr", repr(lr), "--infimum-lr", repr(infimum_lr),
           "--epochs", str(budget), "--seed", str(seed),
           "--tune-budget", str(tune_budget),
           "--subset-ratio", str(subset_ratio),
           "--batch-size", str(batch_size),
           "--device", device, "--out", out]
    if task == "integral":
        cmd += ["--model", model_name, "--data-root", data_root]
    for key, value in sorted(extra.items()):
        if key == "upper_bound":
            cmd += ["--upper-bound", str(int(value))]
        elif key == "cooldown_frac":
            cmd += ["--cooldown-frac", repr(float(value))]
        else:
            cmd += ["--sched-extra", f"{key}={value!r}" if isinstance(value, float)
                    else f"{key}={value}"]
    return cmd


# ── Job list ─────────────────────────────────────────────────────────────────

def parse_list(text, cast):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(
        description="Tune at a small budget, deploy at a larger one.")
    p.add_argument("--task", default="integral", choices=TASKS)
    p.add_argument("--model", default="DeepONet",
                   help="integral only: DeepONet | TFONet")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    p.add_argument("--schedulers", default=",".join(SCHEDULERS),
                   help="comma-separated scheduler codes")
    p.add_argument("--tune-budgets", default=",".join(str(b) for b in TUNE_BUDGETS))
    p.add_argument("--deploy-budgets", default=",".join(str(b) for b in DEPLOY_BUDGETS))
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    p.add_argument("--n-trials", type=int, default=N_TRIALS)
    p.add_argument("--hpo-seed", type=int, default=HPO_SEED,
                   help="seed used for every HPO trial (deployment uses --seeds)")
    p.add_argument("--subset-ratio", type=float, default=0.2, help="cifar only")
    p.add_argument("--batch-size", type=int, default=None,
                   help=f"default per task: {DEFAULT_BATCH_SIZE}")
    p.add_argument("--data-root", default="data", help="integral only")
    p.add_argument("--out", default=None)
    p.add_argument("--study-db", default=None,
                   help="sqlite path. A value ending in .db is one shared study "
                        "file; anything else is a directory holding one "
                        "{scheduler}.db per scheduler (default).")
    p.add_argument("--shard", default="0/1", help="i/n, split by scheduler")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    model_name = args.model if args.task == "integral" else TASK_MODEL[args.task]
    if args.task == "integral" and model_name not in INTEGRAL_MODELS:
        p.error(f"--model must be one of {sorted(INTEGRAL_MODELS)}")
    stem = f"{args.task}-{model_name}" if args.task == "integral" else args.task
    out = args.out or f"results_extrap/{stem}"
    study_db = args.study_db or f"extrap_studies/{stem}"
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE[args.task]

    schedulers = parse_list(args.schedulers, str)
    tune_budgets = parse_list(args.tune_budgets, int)
    deploy_budgets = parse_list(args.deploy_budgets, int)
    seeds = parse_list(args.seeds, int)

    i, n = (int(x) for x in args.shard.split("/"))
    mine = schedulers[i::n]

    n_hpo = len(mine) * len(tune_budgets)
    n_deploy = n_hpo * len(deploy_budgets) * len(seeds)
    hpo_epochs = args.n_trials * sum(tune_budgets) * len(mine)
    deploy_epochs = n_hpo * len(seeds) * sum(deploy_budgets)

    print(f"extrap {stem}: {len(schedulers)} schedulers total, shard {i}/{n} takes "
          f"{len(mine)} ({','.join(mine)})", flush=True)
    print(f"  tune budgets {tune_budgets}, deploy budgets {deploy_budgets}, "
          f"seeds {seeds}, n_trials {args.n_trials}", flush=True)
    print(f"  {n_hpo} HPO stages ({hpo_epochs} trial-epochs), {n_deploy} deploy runs "
          f"({deploy_epochs} epochs)", flush=True)
    print(f"  out={out}  study_db={study_db}  device={args.device}  "
          f"batch_size={batch_size}", flush=True)

    if args.dry_run:
        for scheduler in mine:
            path = study_path(study_db, scheduler)
            for tune_budget in tune_budgets:
                name = study_name_of(args.task, model_name, scheduler, tune_budget)
                done = completed_trials(path, name)
                print(f"  HPO  {scheduler:<4} B_tune={tune_budget:<4} "
                      f"trials={args.n_trials} (done {done}) study={path}::{name}",
                      flush=True)
                for budget in deploy_budgets:
                    for seed in seeds:
                        tag = curve_tag_parts(args.task, model_name, scheduler,
                                              "<hpo>", "<hpo>", budget, seed,
                                              tune_budget)
                        print(f"    RUN  {scheduler:<4} B_tune={tune_budget:<4} "
                              f"N={budget:<4} seed={seed:<5} -> {out}/{tag}.parquet",
                              flush=True)
        print("  (lr and infimum_lr in the filenames are only known after HPO, so "
              "existing-file skipping cannot be previewed here)", flush=True)
        return

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    wandb.init(mode="disabled")
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading data for {args.task}", flush=True)
    dl_train, dl_val = build_data(args.task, args.subset_ratio, batch_size,
                                  data_root=args.data_root)
    acc = task_acc(args.task)
    print(f"[{time.strftime('%H:%M:%S')}] data ready: {len(dl_train)} train batches, "
          f"{len(dl_val)} val batches ({(time.time()-t0)/60:.1f}m)", flush=True)

    stage = 0
    deployed = 0
    for scheduler in mine:
        try:
            for tune_budget in tune_budgets:
                stage += 1
                path = study_path(study_db, scheduler)
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                name = study_name_of(args.task, model_name, scheduler, tune_budget)
                el = (time.time() - t0) / 60
                print(f"[{time.strftime('%H:%M:%S')}] stage {stage}/{n_hpo} "
                      f"(elapsed {el:.1f}m) HPO {scheduler} at B_tune={tune_budget}",
                      flush=True)
                study = run_hpo(args.task, model_name, scheduler, tune_budget,
                                args.n_trials, dl_train, dl_val, device, acc,
                                args.hpo_seed, f"sqlite:///{path}", name)
                best = study.best_trial
                print(f"  best trial {best.number}: val_loss {best.value:.4e} "
                      f"params {best.params}", flush=True)

                for budget in deploy_budgets:
                    lr, infimum_lr, extra = deploy_params(
                        scheduler, best.params, budget, len(dl_train))
                    for seed in seeds:
                        deployed += 1
                        cmd = deploy_cmd(args.task, model_name, scheduler, lr,
                                         infimum_lr, extra, budget, seed,
                                         tune_budget, out, device,
                                         args.subset_ratio, batch_size,
                                         args.data_root)
                        el = (time.time() - t0) / 60
                        print(f"[{time.strftime('%H:%M:%S')}] deploy "
                              f"{deployed}/{n_deploy} (elapsed {el:.1f}m) "
                              + " ".join(cmd), flush=True)
                        r = subprocess.run(cmd)
                        if r.returncode != 0:
                            print(f"  FAILED (exit {r.returncode}), continuing",
                                  flush=True)
        except Exception as exc:
            print(f"  SCHEDULER {scheduler} FAILED: {type(exc).__name__}: {exc}, "
                  f"continuing", flush=True)

    print(f"shard {i}/{n} complete in {(time.time()-t0)/60:.1f} min "
          f"({stage} HPO stages, {deployed} deploy runs)", flush=True)


if __name__ == "__main__":
    main()
