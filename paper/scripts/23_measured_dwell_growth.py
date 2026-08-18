"""
Measured dwell growth: does the time a run actually spends above its own stability boundary grow
with the training budget, and by how much compared with the closed-form prediction?

scripts/19 computes the dwell time from the schedule formula alone. This script measures it from
training: for the SAME tuned configuration deployed at two budgets, count the epochs whose
step size exceeds the measured boundary. The theory says the count scales with the budget for
rescaling families and stays flat for the budget-insensitive ones.

This is a separate claim from "exposure predicts damage", which scripts/20 refutes on this
population (F24). The dwell growth can hold while the damage link fails.

Boundary: Adam-family, 38 / eta on the preconditioned Hessian (Cohen et al., arXiv:2207.14484).
Only epochs up to the loss minimum are counted, since curvature after a collapse is a consequence
of the collapse.

Usage: python 23_measured_dwell_growth.py --dir results_sharp1/integral-DeepONet
"""
import argparse
import collections
import glob
import os

import numpy as np
import polars as pl

# Closed-form top-band sensitivity from scripts/19 (dT_theta/dN at theta = 0.9 eta_0).
PREDICTED_TOP = {"N": 1.000, "WSD": 0.712, "C": 0.205, "L": 0.100, "D2Z": 0.100,
                 "P": 0.081, "H": 0.081, "EH": 0.008, "POW": 0.001, "E": 0.000,
                 "WSQD": 0.000}


def measure(path):
    df = pl.read_parquet(path)
    if "lambda_max_precond" not in df.columns or "lr_pre_step" not in df.columns:
        return None
    lam = df["lambda_max_precond"].to_numpy().astype(float)
    lr = df["lr_pre_step"].to_numpy().astype(float)
    vl = df["val_loss"].to_numpy().astype(float)
    ratio = lr * lam / 38.0
    argmin = int(np.nanargmin(vl))
    pre = np.zeros_like(ratio, dtype=bool)
    pre[:argmin + 1] = True
    ok = np.isfinite(ratio) & pre
    if not ok.any():
        return None
    return {
        "sched": df["scheduler"][0],
        "seed": int(df["seed"][0]),
        "N": int(df["budget"][0]),
        "exp_pre": int((ratio[ok] > 1.0).sum()),
        "maxr_pre": float(np.nanmax(ratio[ok])),
        "argmin": argmin + 1,
        "damage": float(vl[-1] / np.nanmin(vl)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_sharp1/integral-DeepONet")
    ap.add_argument("--low", type=int, default=50)
    ap.add_argument("--high", type=int, default=200)
    args = ap.parse_args()

    rows = [r for r in (measure(p) for p in sorted(glob.glob(os.path.join(args.dir, "*.parquet"))))
            if r is not None]
    cells = collections.defaultdict(dict)
    for r in rows:
        cells[(r["sched"], r["seed"])][r["N"]] = r
    print(f"{len(rows)} runs, {len(cells)} (scheduler, seed) cells in {args.dir}")

    print(f"\nmeasured dwell above the boundary, N={args.low} vs N={args.high}, "
          f"pre-minimum window only")
    print(f"{'sched':<6}{'seed':>6}{'exp(' + str(args.low) + ')':>9}"
          f"{'exp(' + str(args.high) + ')':>10}{'growth':>9}"
          f"{'maxr(' + str(args.low) + ')':>11}{'maxr(' + str(args.high) + ')':>12}"
          f"{'damage(' + str(args.high) + ')':>14}")
    per_sched = collections.defaultdict(list)
    for key in sorted(cells):
        d = cells[key]
        if args.low not in d or args.high not in d:
            continue
        lo, hi = d[args.low], d[args.high]
        g = hi["exp_pre"] / lo["exp_pre"] if lo["exp_pre"] > 0 else float("nan")
        per_sched[key[0]].append((lo["exp_pre"], hi["exp_pre"], g))
        print(f"{key[0]:<6}{key[1]:>6}{lo['exp_pre']:>9}{hi['exp_pre']:>10}"
              f"{g:>9.2f}{lo['maxr_pre']:>11.3f}{hi['maxr_pre']:>12.3f}"
              f"{hi['damage']:>14.3f}")

    print(f"\nper scheduler: summed dwell over seeds, and the closed-form top-band prediction")
    print(f"{'sched':<6}{'sum exp(' + str(args.low) + ')':>13}"
          f"{'sum exp(' + str(args.high) + ')':>14}{'ratio':>8}"
          f"{'predicted dT/dN':>17}")
    tab = []
    for s in sorted(per_sched, key=lambda s: -PREDICTED_TOP.get(s, -1)):
        lo = sum(v[0] for v in per_sched[s])
        hi = sum(v[1] for v in per_sched[s])
        ratio = hi / lo if lo > 0 else float("nan")
        pred = PREDICTED_TOP.get(s, float("nan"))
        tab.append((s, lo, hi, ratio, pred))
        print(f"{s:<6}{lo:>13}{hi:>14}{ratio:>8.2f}{pred:>17.3f}")

    good = [(t[3], t[4]) for t in tab if np.isfinite(t[3])]
    if len(good) >= 3:
        x = np.array([g[1] for g in good])
        y = np.array([g[0] for g in good])
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        print(f"\n  predicted top-band sensitivity vs measured dwell ratio: "
              f"pearson {float(np.corrcoef(x, y)[0, 1]):+.4f}, "
              f"spearman {float(np.corrcoef(rx, ry)[0, 1]):+.4f} over {len(good)} schedulers")
    tot_lo = sum(t[1] for t in tab)
    tot_hi = sum(t[2] for t in tab)
    print(f"  pooled: {tot_lo} epochs above boundary at N={args.low}, {tot_hi} at N={args.high} "
          f"(x{tot_hi / tot_lo:.2f}) while the budget grew x{args.high / args.low:.1f}")


if __name__ == "__main__":
    main()
