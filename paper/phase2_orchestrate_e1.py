"""
E1 orchestrator: the pruning kill-rate grid, run to FULL budget with no pruning.

Why full budget: kill-rate is "of the configurations that low-budget median pruning
would discard, how many would have been top-k at full budget". Answering that needs
the full-budget outcome of the configurations a pruner would have killed, so the grid
must never be pruned. Pruning is then simulated offline by scripts/17_killrate.py.

Grid per task: schedulers x (lr, infimum_lr). The (lr, infimum_lr) values are a strict
superset of the 2026-06 partB grid, so existing partB parquets are reused by the
runner's skip-if-exists rule instead of retrained.

N and SF ignore infimum_lr (constant LR, and the schedule-free optimizer), so they run
the lr axis only.

Sharding: --shard i/n takes every n-th job, so two pueue tasks on two GPUs split the
grid without a shared work list.

Usage:
  python phase2_orchestrate_e1.py --task cifar10 --device cuda:0 --shard 0/2
"""
import argparse
import itertools
import subprocess
import sys
import time

SCHEDULERS = ["N", "C", "WC", "WSD", "L", "P", "H", "WH", "EH", "WEH", "SF"]
NO_INFIMUM = {"N", "SF"}
LRS = [1e-4, 3e-4, 6e-4, 1e-3, 2e-3, 3e-3, 6e-3, 1e-2]
INFS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
INF_DEFAULT = 1e-6
EPOCHS = 100
SEED = 42


def jobs(task):
    out = []
    for sched in SCHEDULERS:
        infs = [INF_DEFAULT] if sched in NO_INFIMUM else INFS
        for lr, inf in itertools.product(LRS, infs):
            out.append((task, sched, lr, inf))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["cifar10", "cifar100", "osc"])
    p.add_argument("--device", default="auto")
    p.add_argument("--shard", default="0/1", help="i/n")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--subset-ratio", type=float, default=0.2)
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    i, n = (int(x) for x in args.shard.split("/"))
    out = args.out or f"results_e1/{args.task}"
    all_jobs = jobs(args.task)
    mine = all_jobs[i::n]
    print(f"E1 {args.task}: {len(all_jobs)} jobs total, shard {i}/{n} takes {len(mine)}, "
          f"device={args.device}, out={out}", flush=True)
    if args.dry_run:
        for j in mine:
            print("  ", j)
        return

    t0 = time.time()
    for k, (task, sched, lr, inf) in enumerate(mine, 1):
        cmd = [sys.executable, "phase2_runner.py",
               "--task", task, "--scheduler", sched,
               "--lr", repr(lr), "--infimum-lr", repr(inf),
               "--epochs", str(args.epochs), "--seed", str(args.seed),
               "--subset-ratio", str(args.subset_ratio),
               "--device", args.device, "--out", out]
        el = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] ({k}/{len(mine)}, elapsed {el/60:.1f}m) "
              + " ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode}), continuing", flush=True)
    print(f"shard {i}/{n} complete in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
