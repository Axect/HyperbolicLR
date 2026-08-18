"""Analyze complete v13 confirmatory artifacts under the frozen statistical contract."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, kendalltau, norm

from confirmatory_orchestrator import select_candidate
from confirmatory_protocol import (
    CORE_SCHEDULERS,
    HORIZONS,
    TASK_SCORE_DIRECTIONS,
    content_hash,
    load_manifest,
    replicate_count,
)


def numeric_metric(value: Any, direction: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value in {"inf", "+inf"}:
        return math.inf
    if value == "-inf":
        return -math.inf
    if direction == "minimize":
        return math.inf
    return -math.inf


def load_summaries(root: Path, stage: str, setting: str) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / stage / setting).glob("*/*.summary.json"))
    ]


def metric_better_key(value: float, direction: str) -> float:
    return value if direction == "minimize" else -value


def exact_sign_flip_pvalue(centered: list[float]) -> float | None:
    if not centered or not all(math.isfinite(value) for value in centered):
        return None
    observed = sum(centered)
    split = len(centered) // 2

    def signed_sums(values: list[float]) -> np.ndarray:
        sums = np.array([0.0])
        for value in values:
            sums = np.concatenate((sums + value, sums - value))
        return np.sort(sums)

    left = signed_sums(centered[:split])
    right = signed_sums(centered[split:])
    count = 0
    tolerance = max(1.0, abs(observed)) * 1e-12
    right_values = right.tolist()
    for value in left:
        index = bisect.bisect_left(right_values, observed - value - tolerance)
        count += len(right_values) - index
    return count / float(2 ** len(centered))


def holm_adjust(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    finite = sorted(
        ((name, value) for name, value in pvalues.items() if value is not None),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float | None] = {name: None for name in pvalues}
    running = 0.0
    count = len(finite)
    for rank, (name, value) in enumerate(finite):
        running = max(running, (count - rank) * value)
        adjusted[name] = min(1.0, running)
    return adjusted


def hierarchical_bca_interval(
    clusters: list[list[float]],
    rng: np.random.Generator,
    repetitions: int = 20_000,
) -> tuple[float, float] | None:
    if len(clusters) < 2 or not all(
        values and all(math.isfinite(value) for value in values) for values in clusters
    ):
        return None
    observed = float(np.mean([np.mean(values) for values in clusters]))
    draws = np.empty(repetitions)
    cluster_count = len(clusters)
    for index in range(repetitions):
        sampled_clusters = rng.integers(0, cluster_count, size=cluster_count)
        cluster_means = []
        for cluster_index in sampled_clusters:
            values = np.asarray(clusters[int(cluster_index)], dtype=float)
            sampled_values = rng.choice(values, size=len(values), replace=True)
            cluster_means.append(float(np.mean(sampled_values)))
        draws[index] = float(np.mean(cluster_means))

    less = np.mean(draws < observed)
    z0 = norm.ppf(np.clip(less, 1.0 / (2 * repetitions), 1 - 1.0 / (2 * repetitions)))
    jackknife = np.array(
        [
            np.mean(
                [
                    np.mean(values)
                    for other, values in enumerate(clusters)
                    if other != omitted
                ]
            )
            for omitted in range(cluster_count)
        ]
    )
    jackknife_mean = float(np.mean(jackknife))
    numerator = float(np.sum((jackknife_mean - jackknife) ** 3))
    denominator = 6.0 * float(np.sum((jackknife_mean - jackknife) ** 2)) ** 1.5
    acceleration = numerator / denominator if denominator else 0.0

    quantiles = []
    for alpha in (0.025, 0.975):
        z_alpha = norm.ppf(alpha)
        adjusted = norm.cdf(z0 + (z0 + z_alpha) / (1 - acceleration * (z0 + z_alpha)))
        quantiles.append(float(np.clip(adjusted, 0.0, 1.0)))
    low, high = np.quantile(draws, quantiles)
    return float(low), float(high)


def paired_effect(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    task: str,
) -> tuple[list[float], list[list[float]]]:
    direction = TASK_SCORE_DIRECTIONS[task]
    right_by_key = {
        (row["hpo_replicate"], row["deployment_seed"]): row for row in right
    }
    by_replicate: dict[int, list[float]] = defaultdict(list)
    for row in left:
        key = (row["hpo_replicate"], row["deployment_seed"])
        comparator = right_by_key[key]
        left_value = numeric_metric(row["test_metric"], direction)
        right_value = numeric_metric(comparator["test_metric"], direction)
        if direction == "minimize":
            delta = math.log10(right_value) - math.log10(left_value)
        else:
            delta = 100.0 * (left_value - right_value)
        by_replicate[int(row["hpo_replicate"])].append(delta)
    ordered = [by_replicate[index] for index in sorted(by_replicate)]
    cluster_scores = [float(np.median(values)) for values in ordered]
    return cluster_scores, ordered


def comparison_margin(protocol: dict[str, Any], comparator: str, task: str) -> float:
    if comparator == "H":
        return float(protocol["primary_comparisons"]["EH_vs_H"]["margin"])
    suffix = "regression" if task == "integral" else "classification"
    specification = protocol["primary_comparisons"][f"EH_vs_{comparator}_{suffix}"]
    key = "margin_log10_loss" if task == "integral" else "margin_accuracy_points"
    return float(specification[key])


def compare_schedulers(
    protocol: dict[str, Any],
    deployment: list[dict[str, Any]],
    setting: str,
    bootstrap_seed: int,
) -> dict[str, Any]:
    task = protocol["settings"][setting]["task"]
    at_200 = [row for row in deployment if int(row["budget"]) == 200]
    results: dict[str, Any] = {}
    raw_pvalues: dict[str, float | None] = {}
    for offset, comparator in enumerate(("H", "POW", "WSQD")):
        eh = [row for row in at_200 if row["scheduler"] == "EH"]
        other = [row for row in at_200 if row["scheduler"] == comparator]
        cluster_scores, seed_clusters = paired_effect(eh, other, task)
        margin = comparison_margin(protocol, comparator, task)
        centered = [value - margin for value in cluster_scores]
        raw_pvalue = exact_sign_flip_pvalue(centered)
        raw_pvalues[comparator] = raw_pvalue
        finite_clusters = [value for value in cluster_scores if math.isfinite(value)]
        sign_test = (
            binomtest(
                sum(value > margin for value in finite_clusters),
                len(finite_clusters),
                0.5,
                alternative="greater",
            ).pvalue
            if finite_clusters
            else None
        )
        leave_one_out = [
            float(np.mean(cluster_scores[:index] + cluster_scores[index + 1 :]))
            for index in range(len(cluster_scores))
            if len(cluster_scores) > 1
        ]
        interval = hierarchical_bca_interval(
            seed_clusters, np.random.default_rng(bootstrap_seed + offset)
        )
        results[comparator] = {
            "effect_direction": "positive favors EH",
            "margin": margin,
            "cluster_count": len(cluster_scores),
            "cluster_scores": cluster_scores,
            "mean_effect": (
                float(np.mean(cluster_scores))
                if all(math.isfinite(value) for value in cluster_scores)
                else None
            ),
            "median_effect": (
                float(np.median(cluster_scores))
                if all(math.isfinite(value) for value in cluster_scores)
                else None
            ),
            "hierarchical_bca_95": list(interval) if interval else None,
            "exact_sign_flip_p": raw_pvalue,
            "exact_sign_test_p": float(sign_test) if sign_test is not None else None,
            "leave_one_replicate_out_range": (
                [float(min(leave_one_out)), float(max(leave_one_out))]
                if leave_one_out
                else None
            ),
        }
    adjusted = holm_adjust(raw_pvalues)
    for comparator, value in adjusted.items():
        results[comparator]["holm_adjusted_p"] = value
    return results


def ranking_diagnostics(
    protocol: dict[str, Any],
    output_root: Path,
    deployment: list[dict[str, Any]],
    setting: str,
) -> list[dict[str, Any]]:
    task = protocol["settings"][setting]["task"]
    direction = TASK_SCORE_DIRECTIONS[task]
    rows: list[dict[str, Any]] = []
    for replicate in range(max(replicate_count(name) for name in CORE_SCHEDULERS)):
        schedulers = [
            name for name in CORE_SCHEDULERS if replicate < replicate_count(name)
        ]
        hpo_values: dict[str, float] = {}
        for scheduler in schedulers:
            candidate = select_candidate(
                output_root, protocol, setting, scheduler, replicate
            )
            selected = [
                row
                for row in load_summaries(output_root, "hpo", setting)
                if row["scheduler"] == scheduler
                and int(row["hpo_replicate"]) == replicate
                and int(row["candidate_index"]) == candidate
            ]
            hpo_values[scheduler] = numeric_metric(
                selected[0]["validation_metric"], direction
            )
        for budget in HORIZONS:
            deployment_values = {}
            for scheduler in schedulers:
                selected_rows = [
                    row
                    for row in deployment
                    if row["scheduler"] == scheduler
                    and int(row["hpo_replicate"]) == replicate
                    and int(row["budget"]) == budget
                ]
                deployment_values[scheduler] = float(
                    np.median(
                        [
                            numeric_metric(row["test_metric"], direction)
                            for row in selected_rows
                        ]
                    )
                )
            hpo_order = sorted(
                schedulers,
                key=lambda name: metric_better_key(hpo_values[name], direction),
            )
            deployment_order = sorted(
                schedulers,
                key=lambda name: metric_better_key(deployment_values[name], direction),
            )
            hpo_ranks = {name: index for index, name in enumerate(hpo_order)}
            deployment_ranks = {
                name: index for index, name in enumerate(deployment_order)
            }
            tau = kendalltau(
                [hpo_ranks[name] for name in schedulers],
                [deployment_ranks[name] for name in schedulers],
            ).statistic
            rows.append(
                {
                    "replicate": replicate,
                    "budget": budget,
                    "scheduler_count": len(schedulers),
                    "kendall_tau": float(tau),
                    "top3_overlap": len(set(hpo_order[:3]) & set(deployment_order[:3])),
                }
            )
    return rows


def deployment_summary(
    protocol: dict[str, Any],
    output_root: Path,
    setting: str,
) -> dict[str, Any]:
    rows = load_summaries(output_root, "deploy", setting)
    expected = sum(
        replicate_count(name) * len(HORIZONS) * 3 for name in CORE_SCHEDULERS
    )
    if len(rows) != expected:
        raise RuntimeError(
            f"deployment artifacts incomplete for {setting}: {len(rows)} != {expected}"
        )
    hpo_rows = load_summaries(output_root, "hpo", setting)
    selected_lrs: dict[str, Any] = {}
    for scheduler in CORE_SCHEDULERS:
        selected_lrs[scheduler] = []
        for replicate in range(replicate_count(scheduler)):
            candidate = select_candidate(
                output_root, protocol, setting, scheduler, replicate
            )
            hpo = next(
                row
                for row in hpo_rows
                if row["scheduler"] == scheduler
                and int(row["hpo_replicate"]) == replicate
                and int(row["candidate_index"]) == candidate
            )
            selected_lrs[scheduler].append(
                {
                    "replicate": replicate,
                    "candidate_index": candidate,
                    "lr": hpo["lr0"],
                }
            )
    return {
        "selected_lrs": selected_lrs,
        "comparisons_at_200": compare_schedulers(
            protocol,
            rows,
            setting,
            protocol["master_seed"] + 10_000 * int(setting[1:]),
        ),
        "ranking_diagnostics": ranking_diagnostics(
            protocol, output_root, rows, setting
        ),
        "eh_nonfinite_count": sum(
            row["scheduler"] == "EH" and row["method_outcome"] == "nonfinite"
            for row in rows
        ),
    }


def candidate_metric(
    rows: list[dict[str, Any]],
    candidate: int,
    budget: int,
    field: str,
    direction: str,
    deployment_seeds: set[int],
) -> float:
    values = [
        numeric_metric(row[field], direction)
        for row in rows
        if int(row["candidate_index"]) == candidate
        and int(row["budget"]) == budget
        and int(row["deployment_seed"]) in deployment_seeds
    ]
    return float(np.median(values))


def arm_a_summary(
    protocol: dict[str, Any], output_root: Path, setting: str = "S1"
) -> dict[str, Any]:
    rows = load_summaries(output_root, "arm-a", setting)
    candidate_count = len(protocol["candidate_grids"]["adamw"]["candidates"])
    expected = len(CORE_SCHEDULERS) * candidate_count * len(HORIZONS) * 3
    if len(rows) != expected:
        raise RuntimeError(f"Arm A incomplete: {len(rows)} != {expected}")
    direction = TASK_SCORE_DIRECTIONS[protocol["settings"][setting]["task"]]
    result: dict[str, Any] = {}
    for scheduler in CORE_SCHEDULERS:
        scheduler_rows = [row for row in rows if row["scheduler"] == scheduler]
        deployment_seeds = sorted(
            {int(row["deployment_seed"]) for row in scheduler_rows}
        )
        entries = []
        for budget in HORIZONS:
            all_seeds = set(deployment_seeds)
            aggregate_metric = partial(
                candidate_metric,
                scheduler_rows,
                direction=direction,
                deployment_seeds=all_seeds,
            )
            short_order = sorted(
                range(candidate_count),
                key=lambda candidate: metric_better_key(
                    aggregate_metric(candidate, 10, "validation_metric"), direction
                ),
            )
            budget_order = sorted(
                range(candidate_count),
                key=lambda candidate: metric_better_key(
                    aggregate_metric(candidate, budget, "validation_metric"), direction
                ),
            )
            short_ranks = {
                candidate: rank for rank, candidate in enumerate(short_order)
            }
            budget_ranks = {
                candidate: rank for rank, candidate in enumerate(budget_order)
            }
            rank_tau = kendalltau(
                [short_ranks[index] for index in range(candidate_count)],
                [budget_ranks[index] for index in range(candidate_count)],
            ).statistic
            folds = []
            for held_out_seed in deployment_seeds:
                fitting_seeds = set(deployment_seeds) - {held_out_seed}
                fit_metric = partial(
                    candidate_metric,
                    scheduler_rows,
                    direction=direction,
                    deployment_seeds=fitting_seeds,
                )
                test_metric = partial(
                    candidate_metric,
                    scheduler_rows,
                    direction=direction,
                    deployment_seeds={held_out_seed},
                )
                selected_10 = min(
                    range(candidate_count),
                    key=lambda candidate: (
                        metric_better_key(
                            fit_metric(candidate, 10, "validation_metric"), direction
                        ),
                        candidate,
                    ),
                )
                oracle = min(
                    range(candidate_count),
                    key=lambda candidate: (
                        metric_better_key(
                            fit_metric(candidate, budget, "validation_metric"),
                            direction,
                        ),
                        candidate,
                    ),
                )
                transferred = test_metric(selected_10, budget, "test_metric")
                oracle_test = test_metric(oracle, budget, "test_metric")
                start_test = test_metric(selected_10, 10, "test_metric")
                if direction == "minimize":
                    regret = math.log10(transferred / oracle_test)
                    utility = math.log10(start_test / transferred)
                else:
                    regret = 100.0 * (oracle_test - transferred)
                    utility = 100.0 * (transferred - start_test)
                folds.append(
                    {
                        "held_out_deployment_seed": held_out_seed,
                        "selected_10_candidate": selected_10,
                        "oracle_candidate": oracle,
                        "transfer_regret": regret,
                        "long_horizon_utility": utility,
                    }
                )
            entries.append(
                {
                    "budget": budget,
                    "transfer_regret": float(
                        np.median([fold["transfer_regret"] for fold in folds])
                    ),
                    "long_horizon_utility": float(
                        np.median([fold["long_horizon_utility"] for fold in folds])
                    ),
                    "candidate_rank_kendall_tau": float(rank_tau),
                    "top3_candidate_overlap": len(
                        set(short_order[:3]) & set(budget_order[:3])
                    ),
                    "cross_fit_folds": folds,
                }
            )
        result[scheduler] = entries
    return result


def write_analysis(
    protocol_path: Path,
    output_root: Path,
    settings: tuple[str, ...],
    include_arm_a: bool,
    output: Path,
) -> dict[str, Any]:
    protocol = load_manifest(protocol_path)
    result: dict[str, Any] = {
        "protocol_hash": protocol["protocol_hash"],
        "settings": {},
    }
    for setting in settings:
        result["settings"][setting] = deployment_summary(protocol, output_root, setting)
    if include_arm_a:
        result["arm_a_S1"] = arm_a_summary(protocol, output_root)
    result["analysis_hash"] = content_hash(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--include-arm-a", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = tuple(item for item in args.settings.split(",") if item)
    result = write_analysis(
        args.protocol,
        args.output_root,
        settings,
        args.include_arm_a,
        args.output,
    )
    print(
        f"wrote {args.output} settings={len(settings)} "
        f"analysis_hash={result['analysis_hash']}"
    )


if __name__ == "__main__":
    main()
