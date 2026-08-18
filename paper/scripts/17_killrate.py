"""
17_killrate.py - does a partial learning curve predict the full-budget ranking?

Two statistics, both taken within one scheduler and never across schedulers: the
question is how faithful a partial-curve HPO decision is for one schedule shape,
not which schedule wins.

1. Early-to-final rank correlation, the primary statistic

    rho(s, e) = spearman(value at epoch e, value at the full budget)

   over the configurations of scheduler s. High rho means the epoch-e ranking
   already is the full-budget ranking, so any low-budget pruner that acts at
   epoch e keeps what a full-budget search would have kept. Low rho means the
   partial curve carries little of the final ranking and a pruner, median or
   ASHA, discards good configurations. rho reads every configuration and every
   rank, so it wastes far less of the data than the top-k count below, which
   collapses a scheduler to k binary hits.

   A scheduler pair is compared with a paired bootstrap over the configuration
   keys the two schedulers share, so that the difference of two rho values does
   not carry the between-configuration variance both of them have in common.

2. Median-pruning kill-rate, kept for continuity with 15_phase2_analysis.py

    kill-rate(s, e, k) = |top_k(s) INTERSECT killed(s, e)| / k

    top_k(s)      = the k configurations of scheduler s that the full budget
                    ranks best
    killed(s, e)  = the configurations of scheduler s that are worse than the
                    median of scheduler s at epoch e

Both statistics are computed for two metrics, selected with --metric:

    val_loss  final validation loss, lower is better (the legacy default)
    acc       final validation accuracy, higher is better

The second metric is not decoration. On CIFAR-10 every schedule reaches its
minimum validation loss well before the budget ends and then overfits, so
"top-k by final validation loss" can be reading which configuration overfits
least rather than which one is best. If the two metrics disagree on the final
ranking, the loss-based statistic is measuring the overfitting tail; --metric
both prints the agreement table that settles it.

--diagnose adds, per scheduler, the spread of the final values, the number of
configurations that never left chance level, and all three tables recomputed with
those configurations removed. Diverged configurations are obvious from epoch one,
so they inflate a rank correlation for free; the divergence-excluded rows are the
honest ones.

Two input sources:

    --parquet-dir  a directory of per-run learning curves (results_phase2/partB
                   layout, columns 'scheduler', 'val_loss', 'acc', 'lr0',
                   'infimum_lr'); one file is one configuration and (task,
                   budget, seed, lr0, infimum_lr) is its key
    --study-db     an Optuna sqlite storage; one trial is one configuration and
                   the curve comes from the trial's intermediate values, which
                   run_all.py records once per epoch. Trials carry no accuracy
                   curve and no configuration key shared between schedulers, so
                   the accuracy metric and the paired test are reported as
                   skipped for this source rather than faked.

Config-resampling bootstrap gives every interval, because a scheduler here
carries only ~10 configurations and a single swap moves the k=3 kill-rate by
0.33.

Usage:
    uv run --with polars --with numpy --with scipy python 17_killrate.py \
        --parquet-dir ../results_phase2/partB
    uv run --with polars --with numpy --with scipy python 17_killrate.py \
        --parquet-dir ../results_phase2/partB --metric both --diagnose
    uv run --with polars --with numpy --with scipy --with optuna python 17_killrate.py \
        --study-db ../HyperbolicLR-CIFAR10-SimpleCNN.db
"""
import argparse
import glob
import itertools
import os
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr


# -- metric conventions ------------------------------------------------------

def positive(v):
    """Validity for a loss: a usable loss is finite and strictly positive."""
    return bool(np.isfinite(v)) and v > 0


def fraction(v):
    """Validity for an accuracy: the `acc` column is stored on a 0..1 scale."""
    return bool(np.isfinite(v)) and 0.0 <= v <= 1.0


@dataclass(frozen=True)
class Metric:
    """One ranking convention: which column, which direction is good.

    `sign` maps the column onto a "smaller is better" scale, so that a single
    kill-rate implementation covers both directions: for the loss the top-k is
    the k smallest final values and the pruner kills what sits above the running
    median, for the accuracy both comparisons flip.
    """

    name: str
    column: str
    title: str
    direction: str
    sign: float
    valid: callable


METRICS = {
    "val_loss": Metric(
        name="val_loss",
        column="val_loss",
        title="final validation loss",
        direction="lower is better",
        sign=1.0,
        valid=positive,
    ),
    "acc": Metric(
        name="acc",
        column="acc",
        title="final validation accuracy",
        direction="higher is better",
        sign=-1.0,
        valid=fraction,
    ),
}

ALL_METRICS = ["val_loss", "acc"]

# Independent bootstrap streams. The kill-rate stream is plain default_rng(seed)
# so that the legacy val_loss kill-rate table is bit-identical to the version of
# this script that carried the kill-rate alone.
STREAM_RANK = 1
STREAM_PAIR = 2
STREAM_DIAG = 3


@dataclass
class Config:
    """One configuration of one scheduler: its curves plus what identifies it.

    `key` has to be comparable across schedulers, because the paired test needs
    the configurations two schedulers have in common. `curves` maps a metric name
    to its per-epoch array, or omits the metric when the source does not carry it.
    """

    sched: str
    key: tuple
    label: str
    curves: dict = field(default_factory=dict)


def aligned(configs, metric, epoch):
    """Values of one scheduler at `epoch` and at the full budget.

    A configuration is dropped for this epoch when its curve is shorter than the
    epoch or when either of the two values is not usable for the metric. The
    top-k and the pruned set are then always taken over the same population, and
    the two arrays returned are index-aligned with the surviving Config list.
    """
    finals, partials, kept = [], [], []
    for c in configs:
        curve = c.curves.get(metric.name)
        if curve is None or len(curve) < epoch:
            continue
        final, partial = float(curve[-1]), float(curve[epoch - 1])
        if not (metric.valid(final) and metric.valid(partial)):
            continue
        finals.append(final)
        partials.append(partial)
        kept.append(c)
    return np.array(finals, dtype=float), np.array(partials, dtype=float), kept


def final_values(configs, metric):
    """Full-budget value of every configuration that carries a usable one."""
    out, kept = [], []
    for c in configs:
        curve = c.curves.get(metric.name)
        if curve is None or len(curve) == 0:
            continue
        final = float(curve[-1])
        if not metric.valid(final):
            continue
        out.append(final)
        kept.append(c)
    return np.array(out, dtype=float), kept


def percentile_ci(samples, ci):
    """Percentile interval that tolerates the resamples where the statistic is undefined."""
    good = samples[~np.isnan(samples)]
    n_bad = int(len(samples) - len(good))
    if len(good) == 0:
        return float("nan"), float("nan"), n_bad
    lo = (100.0 - ci) / 2.0
    return (float(np.percentile(good, lo)),
            float(np.percentile(good, 100.0 - lo)),
            n_bad)


# -- kill-rate ---------------------------------------------------------------

def kill_rate(finals, partials, k, sign=1.0):
    """Kill-rate for one scheduler at one epoch and one k.

    'finals' and 'partials' are aligned arrays over the configurations of a
    single scheduler: the value over the full budget, and the value at the
    pruning epoch. `sign` is the metric's orientation, so that top-k always
    means "the k best" and killed always means "worse than the median".
    """
    n = len(finals)
    if n < k or n == 0:
        return float("nan")
    topk = set(np.argsort(sign * finals, kind="stable")[:k].tolist())
    vals = sign * partials
    killed = set(np.flatnonzero(vals > np.median(vals)).tolist())
    return len(topk & killed) / k


def kill_rate_ci(finals, partials, k, n_boot, rng, ci, sign=1.0):
    """Percentile bootstrap CI over resampled configurations."""
    n = len(finals)
    if n < k or n == 0 or n_boot <= 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = np.array([kill_rate(finals[row], partials[row], k, sign) for row in idx])
    lo, hi, _ = percentile_ci(samples, ci)
    return lo, hi


def scheduler_table(configs, metric, epochs, ks, n_boot, rng, ci):
    """Kill-rate plus CI for every (epoch, k) of one scheduler.

    Configurations whose curve is shorter than the pruning epoch are dropped for
    that epoch only, and the top-k is recomputed on the same surviving subset so
    that the two sets being intersected always cover the same configurations.
    """
    out = {}
    for e in epochs:
        finals, partials, kept = aligned(configs, metric, e)
        for k in ks:
            point = kill_rate(finals, partials, k, metric.sign)
            lo, hi = kill_rate_ci(finals, partials, k, n_boot, rng, ci, metric.sign)
            out[(e, k)] = (point, lo, hi, len(kept))
    return out


# -- early-to-final rank correlation -----------------------------------------

def rank_corr(partials, finals):
    """Spearman rho between the epoch-e ranking and the full-budget ranking.

    Undefined below three configurations, and undefined when one side is
    constant, which is what a bootstrap resample that happens to draw only the
    diverged configurations looks like: they all sit at the same chance-level
    loss. The constant case is caught here rather than left to scipy so that the
    run does not emit a ConstantInputWarning per resample.
    """
    n = len(finals)
    if n < 3:
        return float("nan")
    if np.all(partials == partials[0]) or np.all(finals == finals[0]):
        return float("nan")
    return float(spearmanr(partials, finals)[0])


def rank_corr_ci(partials, finals, n_boot, rng, ci):
    """Percentile bootstrap CI for rho over resampled configurations."""
    n = len(finals)
    if n < 3 or n_boot <= 0:
        return float("nan"), float("nan"), 0
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = np.array([rank_corr(partials[row], finals[row]) for row in idx])
    return percentile_ci(samples, ci)


def rank_table(configs, metric, epochs, n_boot, rng, ci):
    """rho plus CI at every pruning epoch of one scheduler."""
    out = {}
    for e in epochs:
        finals, partials, kept = aligned(configs, metric, e)
        point = rank_corr(partials, finals)
        lo, hi, n_bad = rank_corr_ci(partials, finals, n_boot, rng, ci)
        out[e] = (point, lo, hi, len(kept), n_bad)
    return out


def paired_rank_diff(cfgs_a, cfgs_b, metric, epoch, n_boot, rng, ci):
    """Paired bootstrap CI for rho_a(e) - rho_b(e).

    Only the configuration keys both schedulers carry take part, because the two
    schedulers do not always cover the same grid: in Part B the hyperbolic
    schedulers are missing the two largest-infimum cells that the cosine and WSD
    schedulers have. One resample of that shared key set is applied to both
    schedulers at once, which removes the variance the two rho values share and
    is the only reason a 10-configuration difference can separate at all.

    Returns None when fewer than three keys are shared, since rho is undefined
    there. The rho values reported are the ones restricted to the shared set,
    which is what the difference is actually built from.
    """
    fa, pa, ka = aligned(cfgs_a, metric, epoch)
    fb, pb, kb = aligned(cfgs_b, metric, epoch)
    pos_a = {c.key: i for i, c in enumerate(ka)}
    pos_b = {c.key: i for i, c in enumerate(kb)}
    # Load order, not set order, so the resample indices are reproducible.
    shared = [c.key for c in ka if c.key in pos_b]
    n = len(shared)
    if n < 3:
        return None
    sel_a = np.array([pos_a[key] for key in shared])
    sel_b = np.array([pos_b[key] for key in shared])
    fa, pa = fa[sel_a], pa[sel_a]
    fb, pb = fb[sel_b], pb[sel_b]
    rho_a = rank_corr(pa, fa)
    rho_b = rank_corr(pb, fb)
    point = rho_a - rho_b
    if n_boot <= 0:
        return dict(n=n, rho_a=rho_a, rho_b=rho_b, point=point,
                    lo=float("nan"), hi=float("nan"), n_bad=0)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = np.array([rank_corr(pa[row], fa[row]) - rank_corr(pb[row], fb[row])
                        for row in idx])
    lo, hi, n_bad = percentile_ci(samples, ci)
    return dict(n=n, rho_a=rho_a, rho_b=rho_b, point=point, lo=lo, hi=hi, n_bad=n_bad)


# -- divergence label --------------------------------------------------------

def flag_key(config):
    """Identity of a configuration *within* its scheduler.

    Config.key is deliberately scheduler-independent, because the paired test has
    to recognise the same hyperparameter cell under two schedulers. A per-run
    label therefore has to carry the scheduler as well, or the same cell trained
    under four schedulers collapses onto one entry and only the last one survives.
    """
    return (config.sched, config.key)


def divergence_flags(by_sched, threshold):
    """Configurations whose accuracy curve ends at or below `threshold`.

    Returns None when no configuration carries an accuracy curve. Divergence is
    deliberately not inferred from a loss curve: a loss that plateaus high is
    not distinguishable from a task whose loss scale simply is high, whereas a
    final accuracy at chance level is unambiguous.
    """
    if not any("acc" in c.curves for cfgs in by_sched.values() for c in cfgs):
        return None
    flags = {}
    for cfgs in by_sched.values():
        for c in cfgs:
            curve = c.curves.get("acc")
            if curve is None or len(curve) == 0 or not np.isfinite(curve[-1]):
                flags[flag_key(c)] = False
                continue
            flags[flag_key(c)] = bool(float(curve[-1]) <= threshold)
    return flags


def drop_diverged(by_sched, flags):
    """Same scheduler mapping with the diverged configurations removed."""
    return {s: [c for c in cfgs if not flags.get(flag_key(c), False)]
            for s, cfgs in by_sched.items()}


# -- input: parquet curves ---------------------------------------------------

def load_parquet_curves(parquet_dir, task=None):
    import polars as pl

    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files under {parquet_dir}")
    by_sched = {}
    for f in files:
        df = pl.read_parquet(f)
        if task is not None and "task" in df.columns and df["task"][0] != task:
            continue
        sched = df["scheduler"][0]
        curves = {}
        for name in ALL_METRICS:
            column = METRICS[name].column
            if column in df.columns:
                curves[name] = df[column].to_numpy()
        key, label = parquet_key(df, os.path.basename(f))
        by_sched.setdefault(sched, []).append(
            Config(sched=sched, key=key, label=label, curves=curves))
    return by_sched


def parquet_key(df, filename):
    """Scheduler-independent identity of a run, plus a short label for reports.

    (task, budget, seed, lr0, infimum_lr) is what Part B varies inside one
    scheduler, so it is exactly the key two schedulers can be paired on. Floats
    are rounded before they enter the key so that two files written by the same
    generator cannot miss each other on a last-bit difference. Sources that lack
    the hyperparameter columns fall back to the file name, which pairs nothing
    and is reported as such.
    """
    have = df.columns
    if "lr0" in have and "infimum_lr" in have:
        task = df["task"][0] if "task" in have else "?"
        budget = int(df["budget"][0]) if "budget" in have else -1
        seed = int(df["seed"][0]) if "seed" in have else -1
        lr0 = round(float(df["lr0"][0]), 12)
        inf = round(float(df["infimum_lr"][0]), 12)
        return (task, budget, seed, lr0, inf), f"lr{lr0:g}/inf{inf:g}"
    return ("file", filename), filename


# -- input: optuna storage ---------------------------------------------------

def study_label(study_name):
    """Group label for a study created by run_all.py.

    Study names are '<project>-<optimizer>-<scheduler>'. For the LR-free
    optimizers the scheduler slot is always 'N', so the optimizer is the label
    that identifies the schedule shape.
    """
    parts = study_name.split("-")
    if len(parts) < 2:
        return study_name
    optimizer, scheduler = parts[-2], parts[-1]
    if optimizer in ("Adam", "AdamW"):
        return scheduler
    return optimizer


def load_study_curves(db_path, name_filter=None):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{db_path}"
    by_sched = {}
    n_pruned_total = 0
    n_no_curve = 0
    for summary in optuna.get_all_study_summaries(storage):
        if name_filter is not None and name_filter not in summary.study_name:
            continue
        study = optuna.load_study(study_name=summary.study_name, storage=storage)
        label = study_label(summary.study_name)
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.PRUNED:
                n_pruned_total += 1
                continue
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            iv = trial.intermediate_values
            if not iv:
                n_no_curve += 1
                continue
            curve = np.array([iv[s] for s in sorted(iv)], dtype=float)
            # A trial's sampled hyperparameters are its own, so two schedulers
            # never share a key here and the paired test reports itself skipped.
            key = (summary.study_name, trial.number)
            by_sched.setdefault(label, []).append(
                Config(sched=label, key=key, label=f"t{trial.number}",
                       curves={"val_loss": curve}))
    if n_pruned_total:
        print(f"note: {n_pruned_total} pruned trials skipped; their curves stop "
              f"early, so their full-budget rank is unknown and the kill-rate "
              f"below is conditioned on the trials that ran to completion",
              file=sys.stderr)
    if n_no_curve:
        print(f"note: {n_no_curve} completed trials carry no intermediate values "
              f"(run_all.py was invoked with --no-intermediate-report, or the "
              f"trials predate per-epoch reporting)", file=sys.stderr)
    return by_sched


# -- reporting: kill-rate ----------------------------------------------------

def print_compact(tables, epochs, k):
    """Same shape as part_b_killrate in 15_phase2_analysis.py."""
    print(f"{'sched':<6} {'n_cfg':<6} " +
          " ".join(f"prune@{e:<5}" for e in epochs) + f" (top-{k} kill rate)")
    for sched in sorted(tables):
        table = tables[sched]
        n_cfg = table[(epochs[0], k)][3]
        cells = " ".join(f"{table[(e, k)][0]:<11.2f}" for e in epochs)
        print(f"{sched:<6} {n_cfg:<6} {cells}")


def print_full(tables, epochs, ks, ci):
    head = f"{'sched':<6} {'k':<3} " + " ".join(f"{'e=' + str(e):<22}" for e in epochs)
    print(head)
    print("-" * len(head))
    for sched in sorted(tables):
        table = tables[sched]
        for k in ks:
            cells = []
            for e in epochs:
                point, lo, hi, _ = table[(e, k)]
                if np.isnan(point):
                    cells.append(f"{'n/a':<22}")
                else:
                    cells.append(f"{point:.2f} [{lo:.2f}, {hi:.2f}]".ljust(22))
            print(f"{sched:<6} {k:<3} " + " ".join(cells))
    print(f"(point estimate and {ci:.0f}% percentile CI over resampled configurations)")


def warn_uneven_counts(tables, epochs, ks):
    """Flag schedulers whose usable configuration count changes with the epoch.

    That happens when some curves are shorter than a pruning epoch, in which case
    the rows of that scheduler are computed over different populations and are
    not directly comparable across columns.
    """
    k0 = ks[0]
    for sched in sorted(tables):
        counts = {e: tables[sched][(e, k0)][3] for e in epochs}
        if len(set(counts.values())) > 1:
            print(f"note: {sched} has curves shorter than some pruning epoch; "
                  f"usable configs per epoch = "
                  + ", ".join(f"e={e}:{n}" for e, n in counts.items()))


# -- reporting: rank correlation ---------------------------------------------

def rho_cell(point, lo, hi, width=24):
    if np.isnan(point):
        return "n/a".ljust(width)
    if np.isnan(lo):
        return f"{point:+.2f} [n/a]".ljust(width)
    return f"{point:+.2f} [{lo:+.2f}, {hi:+.2f}]".ljust(width)


def print_rank_tables(tables, epochs, ci):
    """Rows are schedulers, columns are pruning epochs, cells are rho with its CI."""
    head = f"{'sched':<6} {'n_cfg':<6} " + " ".join(f"{'e=' + str(e):<24}" for e in epochs)
    print(head)
    print("-" * len(head))
    for sched in sorted(tables):
        table = tables[sched]
        n_cfg = table[epochs[0]][3]
        cells = [rho_cell(*table[e][:3]) for e in epochs]
        print(f"{sched:<6} {n_cfg:<6} " + " ".join(cells))
    print(f"(Spearman rho and {ci:.0f}% percentile CI over resampled configurations; "
          f"rho = +1 means the epoch-e ranking is already the full-budget ranking)")
    bad = [(s, e, tables[s][e][4]) for s in sorted(tables) for e in epochs
           if tables[s][e][4] > 0]
    for sched, e, n_bad in bad:
        print(f"note: {sched} e={e} had {n_bad} resamples where rho is undefined "
              f"(one side constant); they are excluded from the interval")


def print_pair_tests(by_sched, metric, epochs, n_boot, rng, ci):
    """Paired difference of rho for every scheduler pair, then the separating pairs."""
    scheds = sorted(by_sched)
    rows = []
    unpaired = []
    for s1, s2 in itertools.combinations(scheds, 2):
        for e in epochs:
            res = paired_rank_diff(by_sched[s1], by_sched[s2], metric, e,
                                   n_boot, rng, ci)
            if res is None:
                unpaired.append((s1, s2, e))
                continue
            rows.append((s1, s2, e, res))
    if not rows:
        print("no scheduler pair shares at least three configurations at any of these "
              "epochs, so the paired test is skipped for this source")
        if unpaired:
            print("  pairs without a shared configuration set: "
                  + ", ".join(f"{a} vs {b} at e={e}" for a, b, e in unpaired))
        return []

    head = (f"{'pair':<12} {'e':<4} {'n_shared':<9} {'rho_1':<7} {'rho_2':<7} "
            f"{'diff':<7} {'CI(diff)':<18} sep")
    print(head)
    print("-" * len(head))
    for s1, s2, e, r in rows:
        sep = "yes" if separating(r) else "no"
        # Three decimals on the interval: at two, a bound that sits just below zero
        # prints as +0.00 next to sep=no and reads like a contradiction.
        ci_txt = ("n/a" if np.isnan(r["lo"])
                  else f"[{r['lo']:+.3f}, {r['hi']:+.3f}]")
        print(f"{s1 + ' vs ' + s2:<12} {e:<4} {r['n']:<9} "
              f"{r['rho_a']:<+7.2f} {r['rho_b']:<+7.2f} {r['point']:<+7.2f} "
              f"{ci_txt:<18} {sep}")
    print("(paired bootstrap over the shared configuration keys; rho_1 and rho_2 are "
          "restricted to that shared set, so they can differ from the table above)")
    print(f"(sep = the {ci:.0f}% CI of the difference excludes 0)")

    sep_rows = [r for r in rows if separating(r[3])]
    print()
    expected_fp = len(rows) * (1.0 - ci / 100.0)
    print(f"separating pairs (CI of the difference excludes 0), "
          f"{len(sep_rows)} of {len(rows)} tested (pair, epoch) cells:")
    print(f"  no multiplicity correction is applied, so about {expected_fp:.1f} of "
          f"{len(rows)} cells would separate by chance alone")
    if not sep_rows:
        print("  none")
    for s1, s2, e, r in sep_rows:
        # diff is rho(s1) - rho(s2), so a positive diff means s1 is the faithful one.
        better, worse = (s1, s2) if r["point"] > 0 else (s2, s1)
        rho_worse = r["rho_b"] if r["point"] > 0 else r["rho_a"]
        tail = ("; that rho is negative, so at this epoch the pruner of "
                f"{worse} keeps what the full budget ranks worst"
                if rho_worse < 0 else "")
        print(f"  e={e:<4} {s1} vs {s2}: diff {r['point']:+.2f} "
              f"[{r['lo']:+.2f}, {r['hi']:+.2f}] over n={r['n']} shared configs; "
              f"{better} has the higher rho, so the partial curve of {worse} "
              f"({rho_worse:+.2f}) is the less faithful one{tail}")
    if unpaired:
        print("  (pair, epoch) cells skipped for lack of three shared configurations: "
              + ", ".join(f"{a} vs {b} at e={e}" for a, b, e in unpaired))
    return sep_rows


def separating(res):
    if np.isnan(res["lo"]) or np.isnan(res["hi"]):
        return False
    return res["lo"] > 0.0 or res["hi"] < 0.0


# -- reporting: agreement between the two metrics ----------------------------

def print_metric_churn(by_sched, ks, ci, n_boot, rng):
    """Does the full-budget ranking survive the change of metric?

    The loss-based statistics only mean "which configuration is best" if the
    final-loss ranking agrees with the final-accuracy ranking. On a run that
    overfits after its loss minimum the two orders come apart, and then a
    loss-based top-k is a ranking of overfitting tails. rho here is computed on
    the negated loss, so +1 means the two metrics agree.
    """
    loss, acc = METRICS["val_loss"], METRICS["acc"]
    head = (f"{'sched':<6} {'n_cfg':<6} {'rho(-loss, acc)':<24} "
            + " ".join(f"{'top' + str(k) + ' overlap':<15}" for k in ks)
            + f" {'best by loss':<20} {'best by acc':<20}")
    print(head)
    print("-" * len(head))
    for sched in sorted(by_sched):
        cfgs = by_sched[sched]
        # One population for both metrics, so the two orders are comparable.
        usable = [c for c in cfgs
                  if c.curves.get("val_loss") is not None
                  and c.curves.get("acc") is not None
                  and loss.valid(float(c.curves["val_loss"][-1]))
                  and acc.valid(float(c.curves["acc"][-1]))]
        if not usable:
            print(f"{sched:<6} {'0':<6} {'n/a':<24}")
            continue
        fl = np.array([float(c.curves["val_loss"][-1]) for c in usable])
        fa = np.array([float(c.curves["acc"][-1]) for c in usable])
        point = rank_corr(-fl, fa)
        lo, hi, _ = rank_corr_ci(-fl, fa, n_boot, rng, ci)
        cells = []
        for k in ks:
            if len(usable) < k:
                cells.append(f"{'n/a':<15}")
                continue
            top_loss = set(np.argsort(fl, kind="stable")[:k].tolist())
            top_acc = set(np.argsort(-fa, kind="stable")[:k].tolist())
            cells.append(f"{len(top_loss & top_acc)}/{k}".ljust(15))
        best_loss = usable[int(np.argsort(fl, kind="stable")[0])].label
        best_acc = usable[int(np.argsort(-fa, kind="stable")[0])].label
        print(f"{sched:<6} {len(usable):<6} {rho_cell(point, lo, hi):<24} "
              + " ".join(cells)
              + f" {best_loss:<20} {best_acc:<20}")
    print("(rho is between the negated final loss and the final accuracy, so +1 means "
          "the two metrics rank the configurations the same way)")
    print("(overlap is |top-k by final loss INTERSECT top-k by final accuracy|)")


# -- reporting: diagnostics --------------------------------------------------

def print_spread(by_sched, metric, flags):
    """Spread of the full-budget values, and how many configurations diverged."""
    head = (f"{'sched':<6} {'n_cfg':<6} {'min':<10} {'median':<10} {'max':<10} "
            f"{'n_diverged':<11} diverged configs")
    print(head)
    print("-" * len(head))
    for sched in sorted(by_sched):
        vals, kept = final_values(by_sched[sched], metric)
        if len(vals) == 0:
            print(f"{sched:<6} {'0':<6} {'n/a':<10}")
            continue
        if flags is None:
            n_div, labels = float("nan"), ["(no accuracy column)"]
        else:
            div = [c.label for c in kept if flags.get(flag_key(c), False)]
            n_div, labels = len(div), div or ["-"]
        n_div_txt = "n/a" if isinstance(n_div, float) else str(n_div)
        # %g, not %f: the regression tasks carry losses around 1e-5, which %.4f
        # renders as a column of zeros.
        print(f"{sched:<6} {len(vals):<6} {vals.min():<10.4g} "
              f"{float(np.median(vals)):<10.4g} {vals.max():<10.4g} "
              f"{n_div_txt:<11} " + ", ".join(labels))


def print_diagnose(by_sched, metrics, epochs, ks, n_boot, rng, ci, threshold):
    """Spread, divergence count, and both statistics with the diverged configs removed."""
    flags = divergence_flags(by_sched, threshold)
    print("=== diagnostics ===")
    print(f"divergence label: final accuracy <= {threshold:g}, i.e. the run never left "
          f"chance level. It is read from the accuracy column even when the ranking "
          f"metric is the loss, because a high loss plateau alone does not prove "
          f"divergence.")
    print()
    for metric in metrics:
        print(f"(a) spread of the {metric.title} over the configurations")
        print_spread(by_sched, metric, flags)
        print()
    if flags is None:
        print("(b) divergence count: skipped, this source carries no accuracy column")
        print("(c) divergence-excluded statistics: skipped for the same reason")
        return
    total_div = sum(1 for v in flags.values() if v)
    print(f"(b) {total_div} of {len(flags)} configurations diverged; "
          f"they are listed per scheduler above")
    if total_div == 0:
        print("(c) divergence-excluded statistics: identical to the tables above, "
              "nothing to exclude")
        return
    kept = drop_diverged(by_sched, flags)
    print()
    for metric in metrics:
        print(f"(c) rank correlation on {metric.title} with the diverged "
              f"configurations removed")
        tables = {s: rank_table(cfgs, metric, epochs, n_boot, rng, ci)
                  for s, cfgs in sorted(kept.items())}
        print_rank_tables(tables, epochs, ci)
        print()
        print(f"    paired separation test on {metric.title} with the diverged "
              f"configurations removed")
        print_pair_tests(kept, metric, epochs, n_boot, rng, ci)
        print()
        print(f"    median-pruning kill-rate on {metric.title} with the diverged "
              f"configurations removed")
        ktables = {s: scheduler_table(cfgs, metric, epochs, ks, n_boot, rng, ci)
                   for s, cfgs in sorted(kept.items())}
        if 3 in ks:
            print_compact(ktables, epochs, 3)
            print()
        print_full(ktables, epochs, ks, ci)
        print()
    print("a diverged configuration is obvious from the first epochs, so it lifts rho "
          "for free; the rows above are the ones that describe the configurations an "
          "HPO run actually has to tell apart")


# -- driver ------------------------------------------------------------------

def resolve_metrics(requested, by_sched):
    """Metrics the source can actually support, with an explicit note for the rest."""
    names = ALL_METRICS if requested == "both" else [requested]
    out = []
    for name in names:
        metric = METRICS[name]
        present = any(metric.name in c.curves for cfgs in by_sched.values()
                      for c in cfgs)
        if not present:
            print(f"note: metric '{name}' skipped, this source carries no "
                  f"'{metric.column}' curve", file=sys.stderr)
            continue
        out.append(metric)
    return out


def main():
    p = argparse.ArgumentParser(
        description="Early-to-final rank correlation and median-pruning kill-rate, "
                    "per scheduler"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--parquet-dir", type=str, default=None,
                     help="Directory of learning-curve parquet files")
    src.add_argument("--study-db", type=str, default=None,
                     help="Optuna sqlite storage written by run_all.py")
    p.add_argument("--epochs", type=str, default="10,20,30",
                   help="Comma-separated pruning epochs")
    p.add_argument("--ks", type=str, default="1,3,5",
                   help="Comma-separated top-k sizes")
    p.add_argument("--metric", type=str, default="val_loss",
                   choices=["val_loss", "acc", "both"],
                   help="Ranking metric; 'both' adds the metric-agreement table")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Bootstrap resamples over configurations (0 disables)")
    p.add_argument("--ci", type=float, default=95.0,
                   help="Confidence level in percent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--diagnose", action="store_true",
                   help="Add final-value spread, divergence count, and both "
                        "statistics with the diverged configurations removed")
    p.add_argument("--diverged-acc", type=float, default=0.15,
                   help="Final accuracy at or below which a run counts as diverged")
    p.add_argument("--task", type=str, default=None,
                   help="Restrict parquet curves to one task")
    p.add_argument("--study-filter", type=str, default=None,
                   help="Substring a study name must contain")
    args = p.parse_args()

    epochs = [int(x) for x in args.epochs.split(",")]
    ks = [int(x) for x in args.ks.split(",")]

    if args.parquet_dir is not None:
        by_sched = load_parquet_curves(args.parquet_dir, task=args.task)
        source = f"parquet-dir {args.parquet_dir}"
    else:
        by_sched = load_study_curves(args.study_db, name_filter=args.study_filter)
        source = f"study-db {args.study_db}"

    if not by_sched:
        raise SystemExit(f"no curves found in {source}")

    metrics = resolve_metrics(args.metric, by_sched)
    if not metrics:
        raise SystemExit(f"no requested metric is available in {source}")

    # Separate streams per statistic. The kill-rate stream is seeded exactly as
    # the kill-rate-only version of this script was, so its table does not move
    # when the other statistics draw their own resamples.
    rng_kill = np.random.default_rng(args.seed)
    rng_rank = np.random.default_rng([args.seed, STREAM_RANK])
    rng_pair = np.random.default_rng([args.seed, STREAM_PAIR])
    rng_diag = np.random.default_rng([args.seed, STREAM_DIAG])

    for metric in metrics:
        if len(metrics) > 1:
            print(f"##### metric: {metric.name} ({metric.title}, {metric.direction})"
                  f" #####")
            print()
        suffix = "" if metric.name == "val_loss" else f" on {metric.name}"

        print(f"=== early-to-final rank correlation{suffix} ({source}) ===")
        print(f"rho(s, e) = spearman(value at epoch e, {metric.title}) over the "
              f"configurations of one scheduler")
        print("low rho = the partial curve does not carry the full-budget ranking, "
              "so low-budget pruning discards good configurations")
        print()
        tables = {sched: rank_table(cfgs, metric, epochs, args.bootstrap,
                                    rng_rank, args.ci)
                  for sched, cfgs in sorted(by_sched.items())}
        print_rank_tables(tables, epochs, args.ci)
        print()

        print(f"=== paired separation test{suffix} ({source}) ===")
        print_pair_tests(by_sched, metric, epochs, args.bootstrap, rng_pair, args.ci)
        print()

        print(f"=== median-pruning kill-rate{suffix} ({source}) ===")
        print("kill-rate = share of full-budget top-k configs that pruning at "
              "epoch e would discard")
        print()
        ktables = {sched: scheduler_table(cfgs, metric, epochs, ks, args.bootstrap,
                                          rng_kill, args.ci)
                   for sched, cfgs in sorted(by_sched.items())}
        if 3 in ks:
            print_compact(ktables, epochs, 3)
            print()
        print_full(ktables, epochs, ks, args.ci)
        warn_uneven_counts(ktables, epochs, ks)
        print()

    if len(metrics) > 1:
        print(f"=== final-ranking agreement between the metrics ({source}) ===")
        print_metric_churn(by_sched, ks, args.ci, args.bootstrap, rng_rank)
        print()

    if args.diagnose:
        print_diagnose(by_sched, metrics, epochs, ks, args.bootstrap, rng_diag,
                       args.ci, args.diverged_acc)


if __name__ == "__main__":
    main()
