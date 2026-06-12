"""
Phase 2 decision-run analysis over the curve parquets produced by phase2_runner.py.

Part A artifacts -> two kill-criterion measurements:
1. SLCD per scheduler: same seed, budget pairs (50,100), (50,200), (100,200);
   mean over the common window of |L1 - L2| / (L1 + L2) after Savitzky-Golay-style
   smoothing (moving average window 9, mimicking paper/analyze), averaged over seeds.
2. Anytime regret at stop points tau in {50, 100, 200}: loss of the plan-200 run at
   epoch tau vs the oracle = best final loss among ALL budget-tau runs (any scheduler),
   per seed then averaged.

Part B artifacts -> pruning kill-rate per scheduler:
   offline median pruning at epochs {10, 20, 30} over the 12 (lr, infimum) configs;
   kill-rate = fraction of true top-3 (by final loss) configs that the pruner kills.

Usage: python 15_phase2_analysis.py [results_dir] [task]
"""
import glob
import os
import sys

import numpy as np
import polars as pl

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "../results_phase2"
TASK = sys.argv[2] if len(sys.argv) > 2 else "cifar10"
SCHEDULERS = ["N", "C", "WC", "WSD", "L", "H", "EH", "SF"]
BUDGETS = [50, 100, 200]
SEEDS = [7, 42, 1024]


def smooth(x, w=9):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    pad = np.pad(x, (w // 2, w // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")[: len(x)]


def load_curve(part, task, sched, budget, seed):
    pats = glob.glob(os.path.join(RESULTS, part,
                                  f"{task}_{sched}_lr*_N{budget}_s{seed}.parquet"))
    if not pats:
        return None
    df = pl.read_parquet(pats[0])
    return df["val_loss"].to_numpy()


def slcd(a, b):
    T = min(len(a), len(b))
    a, b = smooth(a[:T]), smooth(b[:T])
    return float(np.mean(np.abs(a - b) / (a + b)))


def part_a_slcd():
    print(f"=== Part A: SLCD ({TASK}, lower = more epoch-insensitive) ===")
    pairs = [(50, 100), (50, 200), (100, 200)]
    header = " ".join(f"{p[0]}-{p[1]:<7}" for p in pairs)
    print(f"{'sched':<6} {header} mean")
    rows = []
    for sched in SCHEDULERS:
        vals = []
        for b1, b2 in pairs:
            per_seed = []
            for seed in SEEDS:
                c1 = load_curve("partA", TASK, sched, b1, seed)
                c2 = load_curve("partA", TASK, sched, b2, seed)
                if c1 is not None and c2 is not None:
                    per_seed.append(slcd(c1, c2))
            vals.append(np.mean(per_seed) if per_seed else np.nan)
        rows.append((sched, vals, np.nanmean(vals)))
        cells = " ".join(f"{v:<10.4f}" for v in vals)
        print(f"{sched:<6} {cells} {np.nanmean(vals):.4f}")
    ranked = sorted(rows, key=lambda r: r[2])
    print("ranking:", " < ".join(r[0] for r in ranked))


def part_a_anytime():
    print(f"\n=== Part A: anytime regret at stop points ({TASK}, plan-200 runs) ===")
    oracle = {}
    for tau in BUDGETS:
        per_seed = {}
        for seed in SEEDS:
            cands = []
            for sched in SCHEDULERS:
                c = load_curve("partA", TASK, sched, tau, seed)
                if c is not None:
                    cands.append(c[-1])
            per_seed[seed] = min(cands) if cands else np.nan
        oracle[tau] = per_seed
    print(f"{'sched':<6} " + " ".join(f"tau={t:<8}" for t in BUDGETS) + " mean")
    for sched in SCHEDULERS:
        regs = []
        for tau in BUDGETS:
            per_seed = []
            for seed in SEEDS:
                c = load_curve("partA", TASK, sched, 200, seed)
                if c is not None and len(c) >= tau:
                    per_seed.append(c[tau - 1] / oracle[tau][seed] - 1)
            regs.append(np.mean(per_seed) if per_seed else np.nan)
        cells = " ".join(f"{100*v:<12.1f}" for v in regs)
        print(f"{sched:<6} {cells} {100*np.nanmean(regs):.1f}%")


def part_b_killrate():
    files = glob.glob(os.path.join(RESULTS, "partB", "*.parquet"))
    if not files:
        print("\n(part B not available yet)")
        return
    print("\n=== Part B: offline median-pruning kill-rate (cifar10) ===")
    by_sched = {}
    for f in files:
        df = pl.read_parquet(f)
        sched = df["scheduler"][0]
        by_sched.setdefault(sched, []).append(df["val_loss"].to_numpy())
    print(f"{'sched':<6} {'n_cfg':<6} " +
          " ".join(f"prune@{e:<5}" for e in [10, 20, 30]) + " (top-3 kill rate)")
    for sched, curves in sorted(by_sched.items()):
        finals = np.array([c[-1] for c in curves])
        top3 = set(np.argsort(finals)[:3])
        cells = []
        for e in [10, 20, 30]:
            partial = np.array([c[e - 1] for c in curves])
            killed = set(np.where(partial > np.median(partial))[0])
            kill_rate = len(top3 & killed) / 3
            cells.append(f"{kill_rate:<11.2f}")
        print(f"{sched:<6} {len(curves):<6} " + " ".join(cells))


if __name__ == "__main__":
    part_a_slcd()
    part_a_anytime()
    part_b_killrate()
