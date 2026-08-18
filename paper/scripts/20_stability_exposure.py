"""
Stability exposure: how long a run actually sat above its own instability threshold, and whether
that predicts the damage.

The dwell analysis in scripts/19 uses an arbitrary level theta. This script replaces theta by the
measured threshold. For plain gradient descent on a locally quadratic loss the stability boundary
is eta = 2 / lambda_max (Cohen et al., arXiv:2103.00065, Sec 2). For Adam-family optimizers the
corresponding boundary is 38 / eta on the preconditioned Hessian P^{-1} H (Cohen et al.,
arXiv:2207.14484), which is what `--rule adam` uses with the lambda_max_precond column.

Per run:
    ratio_t   = lr_pre_step_t * lambda_max_t / 2        (rule sgd)
              = lr_pre_step_t * lambda_max_precond_t / 38   (rule adam)
    exposure  = number of epochs with ratio_t > 1
    max_ratio = max_t ratio_t
    damage    = final val_loss / min val_loss           (1.0 means the run never gave back ground)

Sharpness is measured every K epochs, so unmeasured epochs inherit the last measured lambda_max
(piecewise-constant hold). That is stated rather than hidden: with K > 1 the exposure is an
estimate whose resolution is K epochs.

Hypothesis under test: log(damage) grows with exposure, i.e. self-stabilization has a time budget
(Damian et al., arXiv:2209.15594, show the cubic term drives sharpness back below 2/eta; the claim
here is that this repair fails once the exposure is long enough).

Usage: python 20_stability_exposure.py --parquet-dir DIR [--rule sgd|adam|both]
"""
import argparse
import glob
import os

import numpy as np
import polars as pl


def hold_forward(values):
    """Piecewise-constant fill of NaN gaps with the most recent finite value."""
    out = np.array(values, dtype=float)
    last = np.nan
    for i, v in enumerate(out):
        if np.isfinite(v):
            last = v
        else:
            out[i] = last
    return out


RULES = {"sgd": ("lambda_max", 2.0), "adam": ("lambda_max_precond", 38.0)}


def analyse(path, rule):
    col, denom = RULES[rule]
    df = pl.read_parquet(path)
    have = set(df.columns)
    if col not in have or "lr_pre_step" not in have:
        return None
    lam = hold_forward(df[col].to_numpy())
    lr = df["lr_pre_step"].to_numpy().astype(float)
    vl = df["val_loss"].to_numpy().astype(float)
    n_meas = int(np.isfinite(df[col].to_numpy()).sum())
    ratio = lr * lam / denom
    finite = np.isfinite(ratio)
    if not finite.any():
        return None
    vmin = np.nanmin(vl)
    argmin = int(np.nanargmin(vl))
    # Exposure after the loss has already started degrading is a CONSEQUENCE of the damage, not a
    # cause: once a run blows up its curvature explodes too, so a post-minimum ratio of 183 says
    # nothing about why it blew up. The causal statistic is the crossing that happened while the
    # run was still improving, i.e. within epochs [0, argmin].
    pre = np.zeros_like(ratio, dtype=bool)
    pre[:argmin + 1] = True
    rec = {
        "run": os.path.basename(path),
        "sched": df["scheduler"][0],
        "budget": int(df["budget"][0]),
        "epochs": len(df),
        "n_measured": n_meas,
        "exposure": int((ratio[finite] > 1.0).sum()),
        "exp_pre": int((ratio[finite & pre] > 1.0).sum()),
        "max_ratio": float(np.nanmax(ratio[finite])),
        "max_ratio_pre": (float(np.nanmax(ratio[finite & pre]))
                          if (finite & pre).any() else float("nan")),
        "final": float(vl[-1]),
        "min": float(vmin),
        "argmin": argmin + 1,
        # final/min is 1.0 for any monotone run. max/min is worse than useless: for a healthy run
        # the initial loss IS the maximum, so it measures how much the run learned. The damage
        # measure has to be the rise AFTER the best point was reached, which is 1.0 for a monotone
        # run of any shape and large exactly when the run gave back ground it had already won.
        "damage": float(vl[-1] / vmin) if vmin > 0 else float("nan"),
        "rebound": (float(np.nanmax(vl[int(np.nanargmin(vl)):]) / vmin)
                    if vmin > 0 else float("nan")),
    }
    if "tune_budget" in have:
        rec["tune_budget"] = int(df["tune_budget"][0])
    return rec


def report(rows, rule):
    print(f"\n=== stability exposure, rule {rule} "
          f"({'2 / lambda_max' if rule == 'sgd' else '38 / eta on P^-1 H'}) ===")
    if not rows:
        print("  no run carries the required columns "
              f"({RULES[rule][0]}, lr_pre_step); rerun with --measure-sharpness")
        return
    head = (f"{'sched':<6}{'N':>5}{'B':>5}{'meas':>6}{'exposure':>10}{'exp_pre':>9}{'max_ratio':>11}{'maxr_pre':>11}"
            f"{'argmin':>8}{'final':>12}{'min':>12}{'damage':>10}{'rebound':>12}")
    print(head)
    for r in sorted(rows, key=lambda r: (r["sched"], r["budget"])):
        print(f"{r['sched']:<6}{r['budget']:>5}{r.get('tune_budget', 0):>5}{r['n_measured']:>6}"
              f"{r['exposure']:>10}{r['exp_pre']:>9}{r['max_ratio']:>11.3f}{r['max_ratio_pre']:>11.3f}{r['argmin']:>8}"
              f"{r['final']:>12.4e}{r['min']:>12.4e}{r['damage']:>10.3f}"
              f"{r['rebound']:>12.4g}")
    # Correlations use the pre-damage exposure, which is the only causally admissible one.
    x = np.array([r["exp_pre"] for r in rows], dtype=float)
    for key in ("damage", "rebound"):
        y = np.array([r[key] for r in rows], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
        if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
            print(f"\n  exposure vs log10({key}): needs 3+ runs with both quantities varying "
                  f"(have {int(ok.sum())}, exposure range "
                  f"{np.ptp(x[ok]) if ok.sum() else 0:g}, {key} range "
                  f"{np.ptp(y[ok]) if ok.sum() else 0:g})")
            continue
        ly = np.log10(y[ok])
        pear = float(np.corrcoef(x[ok], ly)[0, 1])
        rx = np.argsort(np.argsort(x[ok]))
        ry = np.argsort(np.argsort(ly))
        spear = float(np.corrcoef(rx, ry)[0, 1])
        slope, icept = np.polyfit(x[ok], ly, 1)
        pred = slope * x[ok] + icept
        den = ((ly - ly.mean()) ** 2).sum()
        r2 = 1.0 - ((ly - pred) ** 2).sum() / den if den > 0 else float("nan")
        print(f"\n  exposure vs log10({key}) over {int(ok.sum())} runs: "
              f"pearson {pear:+.4f}, spearman {spear:+.4f}")
        print(f"  fit: log10({key}) = {slope:.5f} * exposure + {icept:.3f}, R^2 = {r2:.4f}")
    for key, cut in (("damage", 2.0), ("rebound", 10.0)):
        bad = [r for r in rows if np.isfinite(r[key]) and r[key] > cut]
        good = [r for r in rows if np.isfinite(r[key]) and r[key] <= cut]
        if bad and good:
            print(f"  {key} > {cut:g}: exposure median "
                  f"{np.median([r['exposure'] for r in bad]):.1f} "
                  f"vs {np.median([r['exposure'] for r in good]):.1f} for the rest "
                  f"({len(bad)} vs {len(good)} runs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--rule", default="both", choices=["sgd", "adam", "both"])
    ap.add_argument("--glob", default="*.parquet")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.parquet_dir, args.glob)))
    print(f"{len(paths)} parquet files in {args.parquet_dir}")
    rules = ["sgd", "adam"] if args.rule == "both" else [args.rule]
    for rule in rules:
        rows = [r for r in (analyse(p, rule) for p in paths) if r is not None]
        report(rows, rule)


if __name__ == "__main__":
    main()
