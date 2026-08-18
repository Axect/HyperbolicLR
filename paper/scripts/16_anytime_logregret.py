"""
Scale-invariant anytime regret over the Phase 2 Part A curve parquets, under three
different notions of "how good was the run at epoch tau".

Motivation 1 (scale). 15_phase2_analysis.py measures anytime regret as the plain loss
ratio L_sched(tau) / L_oracle(tau) - 1.  On regression tasks whose val loss spans
several orders of magnitude (osc: oracle final losses are O(1e-7)), that ratio explodes
to 10^4 - 10^6 and the resulting table is unusable.  This script replaces the ratio by
its logarithm, which is invariant under rescaling of the loss and stays O(1) nats:

    rho_sched(tau) = log L_sched(tau) - log L_oracle(tau)          [nats]

Motivation 2 (what the val loss actually measures on CIFAR-10).  Every Part A CIFAR-10
run reaches its val loss minimum at epoch 12-58 out of 200 and then overfits, so the
val loss AT epoch tau is not a measure of optimization quality: it is dominated by how
far the run has drifted past its own optimum.  SF ends at val loss 3.49 while holding
75.6% accuracy, and EH takes first place under the val-loss criterion while holding the
family's lowest final val loss (1.23) at 68.9% accuracy, 9 points below WC's 78.0%
(seed medians, plan-200).  Two alternative criteria are therefore provided, and
`--metric all` prints the minimax ranking under all three side by side so the size of
that artifact is visible.

Metrics
    val_loss      (default, unchanged)  rho(tau) with
                    L_sched(tau)  = val loss of the plan-200 run at epoch tau,
                    L_oracle(tau) = min over the family of the FINAL val loss of the
                                    budget-tau runs (as in 15_phase2_analysis.py).
    min_val_loss  rho(tau) with the running minimum instead of the endpoint:
                    L_sched(tau)  = min val loss over epochs 1..tau of the plan-200 run,
                    L_oracle(tau) = min over the family of the min-over-epochs val loss
                                    of the budget-tau runs.
                  This is the loss an early-stopped practitioner would actually ship, so
                  it separates optimization speed from overfitting drift.
    acc           additive accuracy gap in percentage points, lower = better:
                    gap(tau) = 100 * (acc_oracle(tau) - acc_sched(tau)),
                    acc_sched(tau)  = accuracy of the plan-200 run at epoch tau,
                    acc_oracle(tau) = MAX over the family of the final accuracy of the
                                      budget-tau runs.
                  Undefined for regression tasks (osc has no `acc` column); those are
                  reported as skipped rather than treated as an error.

L_oracle / acc_oracle is a best-in-class (family) oracle, not a true oracle: it is the
best value reachable at horizon tau by some member of the measured scheduler family
under the one lr/infimum configuration used in Part A.  A scheduler outside the family,
or a different lr, can beat it, so the regret may be negative and regret = 0 does not
certify optimality.

Seeds are aggregated by median with an interquartile range rather than by mean: there
are only 3 seeds (7, 42, 1024) and the per-seed distribution is heavy tailed, so a mean
is dominated by its worst seed.

For the default metric, exp(rho) - 1 is per seed algebraically identical to the ratio
regret used by 15_phase2_analysis.py; the two reports differ only in aggregation (median
of logs here, mean of ratios there).  The script prints the size of that aggregation gap
so the small-rho regime (classification) can be checked against the legacy numbers.

Usage: python 16_anytime_logregret.py [results_dir] [task] [--metric METRIC]
       METRIC in {val_loss, min_val_loss, acc, all}, default val_loss.
"""
import argparse
import glob
import os
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import polars as pl

# Oracle family, identical to 15_phase2_analysis.py so that L_oracle matches.
SCHEDULERS = ["N", "C", "WC", "WSD", "L", "H", "EH", "SF"]
BUDGETS = [50, 100, 200]
SEEDS = [7, 42, 1024]
PLAN = 200


def positive(v):
    """Validity for a loss: a usable loss is finite and strictly positive."""
    return v is not None and np.isfinite(v) and v > 0


def fraction(v):
    """Validity for an accuracy: the `acc` column is stored on a 0..1 scale."""
    return v is not None and np.isfinite(v) and 0.0 <= v <= 1.0


@dataclass(frozen=True)
class Metric:
    """One anytime-regret definition: which column, how to reduce it, how to score it."""

    name: str
    column: str
    title: str          # goes into the table banner
    unit: str           # "nats" or "pp"
    symbol: str         # "rho" or "gap"
    decimals: int
    log_scale: bool     # log-ratio metrics also get the multiplier table
    definition: str
    oracle_desc: str    # "{family}" is filled in with the comma-joined family
    sched_at: Callable[[np.ndarray, int], float]   # (curve, tau) -> scheduler value
    oracle_of: Callable[[np.ndarray], float]       # budget-tau curve -> oracle candidate
    oracle_best: Callable[[List[float]], float]    # candidates -> oracle value
    score: Callable[[float, float], float]         # (value, oracle) -> regret
    valid: Callable[[float], bool]

    def cell(self, med, iqr):
        if np.isnan(med):
            return "n/a"
        return f"{med:+.{self.decimals}f} ({iqr:.{self.decimals}f})"

    def value(self, v):
        return "n/a" if np.isnan(v) else f"{v:+.{self.decimals}f}"


def _log_ratio(value, oracle):
    return float(np.log(value) - np.log(oracle))


def _acc_gap(value, oracle):
    """Percentage points of accuracy given away relative to the family best."""
    return 100.0 * float(oracle - value)


METRICS = {
    "val_loss": Metric(
        name="val_loss",
        column="val_loss",
        title="anytime log-regret",
        unit="nats",
        symbol="rho",
        decimals=3,
        log_scale=True,
        definition="rho(tau) = log L_sched(tau) - log L_oracle(tau)  [nats, lower = better]",
        oracle_desc=("L_oracle = best-in-class (family) oracle: min final loss over the "
                     "budget-tau runs of {{{family}}}"),
        sched_at=lambda c, tau: float(c[tau - 1]),
        oracle_of=lambda c: float(c[-1]),
        oracle_best=min,
        score=_log_ratio,
        valid=positive,
    ),
    "min_val_loss": Metric(
        name="min_val_loss",
        column="val_loss",
        title="anytime log-regret on the running min val loss",
        unit="nats",
        symbol="rho",
        decimals=3,
        log_scale=True,
        definition=("rho(tau) = log min_{e<=tau} L_sched(e) - log L_oracle(tau)  "
                    "[nats, lower = better]"),
        oracle_desc=("L_oracle = best-in-class (family) oracle: min over epochs and over "
                     "the budget-tau runs of {{{family}}}"),
        # A NaN anywhere in epochs 1..tau invalidates the window, so plain min is used
        # instead of nanmin: a run with a broken epoch inside the window is not trusted.
        sched_at=lambda c, tau: float(np.min(c[:tau])),
        oracle_of=lambda c: float(np.min(c)),
        oracle_best=min,
        score=_log_ratio,
        valid=positive,
    ),
    "acc": Metric(
        name="acc",
        column="acc",
        title="anytime accuracy regret",
        unit="pp",
        symbol="gap",
        decimals=2,
        log_scale=False,
        definition=("gap(tau) = acc_oracle(tau) - acc_sched(tau)  "
                    "[percentage points, lower = better]"),
        oracle_desc=("acc_oracle = best-in-class (family) oracle: MAX final accuracy over "
                     "the budget-tau runs of {{{family}}}"),
        sched_at=lambda c, tau: float(c[tau - 1]),
        oracle_of=lambda c: float(c[-1]),
        oracle_best=max,
        score=_acc_gap,
        valid=fraction,
    ),
}

ALL_METRICS = ["val_loss", "min_val_loss", "acc"]


def load_curve(results, part, task, sched, budget, seed, column="val_loss"):
    pats = glob.glob(os.path.join(results, part,
                                  f"{task}_{sched}_lr*_N{budget}_s{seed}.parquet"))
    if not pats:
        return None
    df = pl.read_parquet(pats[0])
    if column not in df.columns:
        return None
    return df[column].to_numpy()


def discover_schedulers(results, task):
    """Every scheduler with at least one Part A curve for `task`, family first."""
    found = set()
    for path in glob.glob(os.path.join(results, "partA", f"{task}_*.parquet")):
        name = os.path.basename(path)[len(task) + 1:]
        found.add(name.split("_lr")[0])
    extra = sorted(found - set(SCHEDULERS))
    return [s for s in SCHEDULERS if s in found], extra


def has_column(results, task, column):
    """True if the Part A parquets for `task` carry `column` at all."""
    for path in sorted(glob.glob(os.path.join(results, "partA", f"{task}_*.parquet"))):
        return column in pl.read_parquet(path).columns
    return False


def oracle_values(results, task, family, metric):
    """oracle[tau][seed]: family-best value reachable at horizon tau."""
    out = {}
    for tau in BUDGETS:
        per_seed = {}
        for seed in SEEDS:
            cands = []
            for sched in family:
                c = load_curve(results, "partA", task, sched, tau, seed, metric.column)
                if c is None or len(c) == 0:
                    continue
                v = metric.oracle_of(c)
                if metric.valid(v):
                    cands.append(v)
            per_seed[seed] = metric.oracle_best(cands) if cands else np.nan
        out[tau] = per_seed
    return out


def collect(results, task, rows, oracle, metric):
    """score[sched][tau] -> per-seed regrets, raw[sched][tau] -> (value, oracle) pairs."""
    score = {s: {t: [] for t in BUDGETS} for s in rows}
    raw = {s: {t: [] for t in BUDGETS} for s in rows}
    # `bad_loss` counts an unusable scheduler value; for the acc metric read it as
    # `bad_value`.  The key name is kept so the default report stays comparable.
    dropped = {"missing_curve": 0, "short_curve": 0, "bad_loss": 0, "bad_oracle": 0}
    for sched in rows:
        for seed in SEEDS:
            c = load_curve(results, "partA", task, sched, PLAN, seed, metric.column)
            if c is None:
                dropped["missing_curve"] += len(BUDGETS)
                continue
            for tau in BUDGETS:
                if len(c) < tau:
                    dropped["short_curve"] += 1
                    continue
                num = metric.sched_at(c, tau)
                den = oracle[tau][seed]
                if not metric.valid(num):
                    dropped["bad_loss"] += 1
                    continue
                if not metric.valid(den):
                    dropped["bad_oracle"] += 1
                    continue
                score[sched][tau].append(metric.score(num, den))
                raw[sched][tau].append((num, den))
    return score, raw, dropped


def med_iqr(vals):
    if not vals:
        return np.nan, np.nan
    a = np.asarray(vals, dtype=float)
    q75, q25 = np.percentile(a, [75, 25])
    return float(np.median(a)), float(q75 - q25)


def report(results, task, metric):
    """Print one metric's tables; return {sched: (minimax, median)} for the comparison."""
    family, extra = discover_schedulers(results, task)
    rows = family + extra
    if not rows:
        print(f"(no Part A curves for task {task} under {results})")
        return None, []
    if metric.column != "val_loss" and not has_column(results, task, metric.column):
        print(f"{task}: task has no {metric.column} column, skipping")
        return None, extra

    oracle = oracle_values(results, task, family, metric)
    score, raw, dropped = collect(results, task, rows, oracle, metric)

    print(f"=== Part A: {metric.title} ({task}, plan-{PLAN} runs) ===")
    print(metric.definition)
    print(metric.oracle_desc.format(family=", ".join(family)))
    if extra:
        print(f"* rows marked with * are outside the oracle family: {', '.join(extra)}"
              f" (their {metric.symbol} can go below 0)")
    print(f"cells: median (IQR) over seeds {SEEDS}")
    print()

    print(f"{'sched':<7}" + "".join(f"tau={t:<16}" for t in BUDGETS))
    stats = {}
    for sched in rows:
        cells, meds = [], []
        for tau in BUDGETS:
            m, i = med_iqr(score[sched][tau])
            meds.append(m)
            cells.append(metric.cell(m, i).ljust(20))
        stats[sched] = meds
        label = f"{sched}*" if sched in extra else sched
        print(f"{label:<7}" + "".join(cells))

    if metric.log_scale:
        print()
        print("same cells as loss multipliers: exp(median rho), spread as the "
              "multiplicative q75/q25 factor exp(IQR)")
        print(f"{'sched':<7}" + "".join(f"tau={t:<16}" for t in BUDGETS))
        for sched in rows:
            cells = []
            for tau in BUDGETS:
                m, i = med_iqr(score[sched][tau])
                if np.isnan(m):
                    cells.append("n/a".ljust(20))
                    continue
                cells.append(f"{np.exp(m):.3f}x (x{np.exp(i):.2f})".ljust(20))
            label = f"{sched}*" if sched in extra else sched
            print(f"{label:<7}" + "".join(cells))

    print()
    print(f"summary over tau in {{50, 100, 200}} of the per-tau median {metric.symbol} "
          f"[{metric.unit}]:")
    header = f"{'sched':<7} {'minimax':<12} {'median':<12}"
    if metric.log_scale:
        header += f" {'minimax as mult':<16}"
    print(header)
    summ = {}
    for sched in rows:
        meds = [m for m in stats[sched] if not np.isnan(m)]
        if not meds:
            continue
        mx, md = float(np.max(meds)), float(np.median(meds))
        summ[sched] = (mx, md)
        label = f"{sched}*" if sched in extra else sched
        line = f"{label:<7} {mx:<+12.{metric.decimals}f} {md:<+12.{metric.decimals}f}"
        if metric.log_scale:
            line += f" {np.exp(mx):<16.3f}"
        print(line)
    ranked = sorted(summ, key=lambda s: summ[s][0])
    print(f"ranking by minimax {metric.symbol}:", " < ".join(
        (f"{s}*" if s in extra else s) for s in ranked))

    n_drop = sum(dropped.values())
    print()
    print(f"excluded cells: {n_drop} "
          + ", ".join(f"{k}={v}" for k, v in dropped.items()))

    if metric.name == "val_loss":
        # Aggregation gap against the legacy ratio regret of 15_phase2_analysis.py.
        # Per seed exp(rho) - 1 == L/L_oracle - 1 exactly, so any gap is aggregation only.
        gaps = []
        for sched in rows:
            for tau in BUDGETS:
                vals = score[sched][tau]
                if not vals:
                    continue
                new = float(np.exp(np.median(vals))) - 1.0
                legacy = float(np.mean([n / d - 1.0 for n, d in raw[sched][tau]]))
                peak = max(abs(v) for v in vals)
                gaps.append((abs(new - legacy), sched, tau, new, legacy, peak))
        small = [g for g in gaps if g[5] < 0.5]
        if small:
            worst = max(small, key=lambda g: g[0])
            print(f"small-rho check (|rho| < 0.5 nats, {len(small)}/{len(gaps)} cells): "
                  f"max |exp(median rho) - 1 minus mean legacy ratio regret| = "
                  f"{worst[0]:.4f} at {worst[1]} tau={worst[2]} "
                  f"({100*worst[3]:.1f}% vs {100*worst[4]:.1f}%)")
        big = [g for g in gaps if g[5] >= 0.5]
        if big:
            worst = max(big, key=lambda g: g[0])
            print(f"large-rho cells ({len(big)}/{len(gaps)}): outside the linear regime, "
                  f"the mean-of-ratios aggregation runs away from the median-of-logs one, "
                  f"worst at {worst[1]} tau={worst[2]}: {100*worst[3]:.1f}% here vs "
                  f"{100*worst[4]:.1f}% legacy")
    return summ, extra


def compare(task, done, extra):
    """Put the per-metric minimax rankings side by side and name the rank churn."""
    names = [n for n, s in done.items() if s]
    if len(names) < 2:
        return
    order = []
    for name in names:
        for sched in done[name]:
            if sched not in order:
                order.append(sched)
    ranks = {}
    for name in names:
        ranked = sorted(done[name], key=lambda s: done[name][s][0])
        ranks[name] = {s: i + 1 for i, s in enumerate(ranked)}

    print(f"=== does the metric choice change the ranking? ({task}) ===")
    print("minimax over tau in {50, 100, 200} of the per-tau median regret, "
          "rank in parentheses; lower = better everywhere")
    print(f"{'sched':<7}" + "".join(
        f"{METRICS[n].name + ' [' + METRICS[n].unit + ']':<22}" for n in names))
    for sched in order:
        cells = []
        for name in names:
            m = METRICS[name]
            if sched not in done[name]:
                cells.append("n/a".ljust(22))
                continue
            cells.append(f"{m.value(done[name][sched][0])} ({ranks[name][sched]})".ljust(22))
        label = f"{sched}*" if sched in extra else sched
        print(f"{label:<7}" + "".join(cells))
    print()
    for name in names:
        ranked = sorted(done[name], key=lambda s: done[name][s][0])
        print(f"{METRICS[name].name:<13} ranking: " + " < ".join(
            (f"{s}*" if s in extra else s) for s in ranked))

    base = names[0]
    for name in names[1:]:
        shifts = []
        for sched in ranks[base]:
            if sched not in ranks[name]:
                continue
            d = ranks[name][sched] - ranks[base][sched]
            if d:
                shifts.append((abs(d), sched, ranks[base][sched], ranks[name][sched]))
        if not shifts:
            print(f"rank shift {base} -> {name}: none")
            continue
        shifts.sort(key=lambda x: -x[0])
        print(f"rank shift {base} -> {name}: " + ", ".join(
            f"{s} {a}->{b}" for _, s, a, b in shifts))


def main():
    ap = argparse.ArgumentParser(
        description="Anytime regret over the Phase 2 Part A curves, "
                    "under a choice of quality metric.")
    ap.add_argument("results_dir", nargs="?", default="../results_phase2",
                    help="directory holding partA/ and partB/ (default ../results_phase2)")
    ap.add_argument("task", nargs="?", default="cifar10",
                    help="task prefix of the Part A parquets (default cifar10)")
    ap.add_argument("--metric", default="val_loss",
                    choices=ALL_METRICS + ["all"],
                    help="quality metric at epoch tau (default val_loss); "
                         "'all' prints every metric plus a ranking comparison")
    args = ap.parse_args()

    if args.metric != "all":
        report(args.results_dir, args.task, METRICS[args.metric])
        return

    done, extra = {}, []
    for i, name in enumerate(ALL_METRICS):
        if i:
            print()
        summ, ex = report(args.results_dir, args.task, METRICS[name])
        done[name] = summ
        extra = ex or extra
    print()
    compare(args.task, done, extra)


if __name__ == "__main__":
    main()
