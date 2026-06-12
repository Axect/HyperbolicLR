"""
Phase 2 decision-run orchestrator: sequential single-GPU experiment grid.
Resume-safe (the runner skips existing parquet outputs), so the loop can be
re-launched after interruption.

Part A (SLCD + anytime regret artifacts):
  tasks {cifar10 (subset 0.2), osc} x schedulers x budgets {50,100,200} x 3 seeds
Part B (pruning kill-rate artifacts):
  cifar10 x schedulers {C, WSD, H, EH} x 12 (lr, infimum) configs x 100 epochs x 1 seed
"""
import itertools
import subprocess
import sys
import time

PY = sys.executable
SCHEDULERS_A = ["N", "C", "WC", "WSD", "L", "H", "EH", "SF"]
BUDGETS = [50, 100, 200]
SEEDS = [7, 42, 1024]
SCHEDULERS_B = ["C", "WSD", "H", "EH"]
LRS_B = [0.0003, 0.001, 0.003, 0.01]
INFS_B = [1e-6, 1e-4, 1e-3]


def run(args):
    cmd = [PY, "phase2_runner.py"] + args
    print(f"[{time.strftime('%H:%M:%S')}]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"  FAILED (exit {r.returncode}), continuing", flush=True)


def part_a():
    for task in ["cifar10", "osc"]:
        for sched, budget, seed in itertools.product(SCHEDULERS_A, BUDGETS, SEEDS):
            args = ["--task", task, "--scheduler", sched, "--seed", str(seed),
                    "--epochs", str(budget), "--out", "results_phase2/partA"]
            if task == "cifar10":
                args += ["--subset-ratio", "0.2", "--lr", "0.003"]
            else:
                args += ["--lr", "0.003"]
            if sched == "SF":
                pass  # SF uses the same base lr for fairness
            run(args)


def part_b():
    for sched, lr, inf in itertools.product(SCHEDULERS_B, LRS_B, INFS_B):
        run(["--task", "cifar10", "--scheduler", sched, "--seed", "42",
             "--epochs", "100", "--lr", str(lr), "--infimum-lr", str(inf),
             "--subset-ratio", "0.2", "--out", "results_phase2/partB"])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "a"):
        part_a()
    if which in ("all", "b"):
        part_b()
    print("orchestration complete")
