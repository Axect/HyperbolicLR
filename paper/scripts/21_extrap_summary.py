"""
Budget-extrapolation summary: take a configuration tuned at budget B and deploy it unchanged at
larger budgets N, then report what the extension cost.

Per (scheduler, tune budget B, seed):
    ratio(N)   = final val loss at N  /  final val loss at the smallest available N
    rebound(N) = final val loss at N  /  min val loss within that same run
                 (1.0 for a monotone run of any shape; > 1 means the run gave back ground)

The prediction under test comes from the dwell-sensitivity spectrum in scripts/19: families whose
sensitivity sits at the top of the level spectrum should degrade when the budget is extended
{C, P, L, D2Z, WSD}, and families that keep it away from the top should not {EH, POW, WSQD, E}.

Usage: python 21_extrap_summary.py [--dir results_extrap/integral-DeepONet]
"""
import argparse
import collections
import glob
import os

import numpy as np
import polars as pl

SAFE = {"EH", "POW", "WSQD", "E"}
RISKY = {"C", "P", "L", "D2Z", "WSD", "WC"}
REBOUND_CUT = 1.1
BROKEN_FACTOR = 10.0


def load(directory):
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*.parquet"))):
        df = pl.read_parquet(path)
        vl = df["val_loss"].to_numpy().astype(float)
        rows.append({
            "sched": df["scheduler"][0],
            "N": int(df["budget"][0]),
            "B": int(df["tune_budget"][0]) if "tune_budget" in df.columns else 0,
            "seed": int(df["seed"][0]),
            "lr0": float(df["lr0"][0]),
            "final": float(vl[-1]),
            "vmin": float(np.nanmin(vl)),
            "argmin": int(np.nanargmin(vl)) + 1,
            "epochs": len(vl),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_extrap/integral-DeepONet")
    args = ap.parse_args()

    rows = load(args.dir)
    print(f"{len(rows)} runs in {args.dir}")
    if not rows:
        return
    cells = collections.defaultdict(dict)
    for r in rows:
        cells[(r["sched"], r["B"], r["seed"])][r["N"]] = r
    budgets = sorted({r["N"] for r in rows})

    print(f"\ndeploy budgets present: {budgets}")
    print(f"{'sched':<6}{'B':>4}{'seed':>6}{'lr0':>10}"
          + "".join(f"{'L(' + str(n) + ')':>12}" for n in budgets)
          + f"{'ratio':>9}{'rebound':>9}")
    per_sched = collections.defaultdict(list)
    for key in sorted(cells):
        d = cells[key]
        base = min(d)
        top = max(d)
        ratio = d[top]["final"] / d[base]["final"] if len(d) > 1 else float("nan")
        reb = d[top]["final"] / d[top]["vmin"]
        cellstr = "".join(f"{d[n]['final']:>12.3e}" if n in d else f"{'-':>12}" for n in budgets)
        print(f"{key[0]:<6}{key[1]:>4}{key[2]:>6}{d[base]['lr0']:>10.2e}{cellstr}"
              f"{ratio:>9.3f}{reb:>9.3f}")
        if len(d) > 1 and np.isfinite(ratio):
            per_sched[key[0]].append(ratio)

    if not per_sched:
        print("\nno scheduler has two deploy budgets yet, so no extrapolation ratio is defined")
        return
    print(f"\nper-scheduler median extrapolation ratio (largest N / smallest N available)")
    print(f"{'sched':<6}{'n_cells':>9}{'median':>10}{'max':>10}{'predicted':>11}")
    for s in sorted(per_sched, key=lambda s: -np.median(per_sched[s])):
        pred = "risky" if s in RISKY else ("safe" if s in SAFE else "n/a")
        v = per_sched[s]
        print(f"{s:<6}{len(v):>9}{np.median(v):>10.3f}{max(v):>10.3f}{pred:>11}")
    risky = [np.median(per_sched[s]) for s in per_sched if s in RISKY]
    safe = [np.median(per_sched[s]) for s in per_sched if s in SAFE]
    if risky and safe:
        print(f"\n  risky group median of medians {np.median(risky):.3f} "
              f"vs safe group {np.median(safe):.3f}")
    else:
        print(f"\n  only one predicted group present so far "
              f"(risky {len(risky)}, safe {len(safe)}); the contrast needs both")
    # The median ratio is contaminated: a cell whose SMALLEST budget was already broken has a
    # flattering ratio (WSD B=10 seed 89 starts at 4.77e-2 and then "improves"). Two clean
    # statistics instead:
    #   collapse       = rebound > REBOUND_CUT at the largest budget, the run gave back ground
    #   broken_at_tune = the smallest-budget loss is an outlier against the same (B, seed) cohort
    print(f"\ncollapse counting (rebound > {REBOUND_CUT}) and runs broken at the tuning budget")
    base_by_cell = collections.defaultdict(list)
    for key, d in cells.items():
        base_by_cell[(key[1], key[2])].append(d[min(d)]["final"])
    med_base = {k: float(np.median(v)) for k, v in base_by_cell.items()}
    stat = collections.defaultdict(lambda: {"n": 0, "collapse": 0, "broken": 0,
                                            "max_reb": 0.0, "worst_ratio": 0.0})
    for key, d in cells.items():
        s, b, seed = key
        top, base = max(d), min(d)
        reb = d[top]["final"] / d[top]["vmin"]
        st = stat[s]
        st["n"] += 1
        st["max_reb"] = max(st["max_reb"], reb)
        if reb > REBOUND_CUT:
            st["collapse"] += 1
        if d[base]["final"] > BROKEN_FACTOR * med_base[(b, seed)]:
            st["broken"] += 1
        if len(d) > 1:
            st["worst_ratio"] = max(st["worst_ratio"], d[top]["final"] / d[base]["final"])
    print(f"{'sched':<6}{'cells':>6}{'collapse':>10}{'broken@tune':>13}{'max rebound':>13}"
          f"{'worst ratio':>13}{'predicted':>11}")
    for s in sorted(stat, key=lambda s: (stat[s]["collapse"], stat[s]["max_reb"])):
        st = stat[s]
        pred = "risky" if s in RISKY else ("safe" if s in SAFE else "n/a")
        print(f"{s:<6}{st['n']:>6}{st['collapse']:>10}{st['broken']:>13}"
              f"{st['max_reb']:>13.4g}{st['worst_ratio']:>13.4g}{pred:>11}")
    clean = sorted(s for s in stat if stat[s]["collapse"] == 0)
    print(f"\n  zero-collapse schedulers: {clean if clean else 'none'}")
    for group, name in ((SAFE, "predicted safe"), (RISKY, "predicted risky")):
        present = sorted(s for s in stat if s in group)
        if present:
            tot = sum(stat[s]["collapse"] for s in present)
            cel = sum(stat[s]["n"] for s in present)
            print(f"  {name}: {tot} collapses over {cel} cells ({', '.join(present)})")


if __name__ == "__main__":
    main()

