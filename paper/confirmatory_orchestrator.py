"""Plan, execute, and audit the frozen v13 production campaign."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from confirmatory_data import load_data_manifest
from confirmatory_protocol import (
    CORE_SCHEDULERS,
    HORIZONS,
    OUTPUT_SCHEMA,
    TASK_SCORE_DIRECTIONS,
    TUNE_BUDGET,
    content_hash,
    file_sha256,
    load_manifest,
    make_run_id,
    make_selection_id,
    metric_token,
    replicate_count,
)

PRODUCTION_PHASES = ("hpo", "deploy", "arm-a")


@dataclass(frozen=True)
class Job:
    stage: str
    setting: str
    scheduler: str
    hpo_replicate: int
    candidate_index: int
    budget: int
    seed_slot: int


def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def parse_csv_phases(value: str) -> tuple[str, ...]:
    values = tuple(item for item in value.split(",") if item)
    invalid = sorted(set(values) - set(PRODUCTION_PHASES))
    if invalid or not values:
        raise argparse.ArgumentTypeError(f"invalid production phases: {invalid}")
    return values


def scheduler_replicates(scheduler: str, mode: str) -> range:
    if mode == "pilot":
        return range(1)
    if mode == "full":
        return range(replicate_count(scheduler))
    raise ValueError(f"unknown replicate mode: {mode}")


def hpo_summaries(
    output_root: Path,
    setting: str,
    scheduler: str,
    replicate: int,
) -> list[dict[str, Any]]:
    root = output_root / "hpo" / setting / scheduler
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in root.glob("*.summary.json")
    ]
    summaries = [row for row in summaries if int(row["hpo_replicate"]) == replicate]
    return sorted(summaries, key=lambda row: int(row["candidate_index"]))


def select_candidate(
    output_root: Path,
    protocol: dict[str, Any],
    setting: str,
    scheduler: str,
    replicate: int,
) -> int:
    setting_spec = protocol["settings"][setting]
    grid = protocol["candidate_grids"][setting_spec["optimizer_key"]]["candidates"]
    summaries = hpo_summaries(output_root, setting, scheduler, replicate)
    if len(summaries) != len(grid):
        raise RuntimeError(
            f"incomplete HPO selection {setting}/{scheduler}/R{replicate}: "
            f"{len(summaries)} of {len(grid)} candidates"
        )
    candidate_indices = [int(row["candidate_index"]) for row in summaries]
    if candidate_indices != list(range(len(grid))):
        raise RuntimeError(
            f"HPO candidate set mismatch for {setting}/{scheduler}/R{replicate}"
        )
    expected_grid_hash = protocol["candidate_grids"][setting_spec["optimizer_key"]][
        "sha256"
    ]
    expected_direction = TASK_SCORE_DIRECTIONS[setting_spec["task"]]
    for row in summaries:
        expected = {
            "protocol_hash": protocol["protocol_hash"],
            "stage": "hpo",
            "setting": setting,
            "scheduler": scheduler,
            "hpo_replicate": replicate,
            "candidate_grid_hash": expected_grid_hash,
            "metric_direction": expected_direction,
            "run_status": "complete",
            "dirty": False,
            "test_accessed": False,
            "test_metric": None,
        }
        mismatched = [key for key, value in expected.items() if row.get(key) != value]
        if mismatched:
            raise RuntimeError(
                f"invalid HPO selection provenance for "
                f"{setting}/{scheduler}/R{replicate}: {mismatched}"
            )
    finite = [
        row
        for row in summaries
        if isinstance(row["validation_metric"], (int, float))
        and math.isfinite(float(row["validation_metric"]))
    ]
    if not finite:
        raise RuntimeError(
            f"all HPO candidates are non-finite for {setting}/{scheduler}/R{replicate}"
        )
    direction = TASK_SCORE_DIRECTIONS[setting_spec["task"]]
    sign = 1.0 if direction == "minimize" else -1.0
    selected = min(
        finite,
        key=lambda row: (
            sign * float(row["validation_metric"]),
            int(row["candidate_index"]),
        ),
    )
    return int(selected["candidate_index"])


def build_jobs(
    protocol: dict[str, Any],
    output_root: Path,
    phase: str,
    setting: str,
    replicate_mode: str,
    horizons: tuple[int, ...],
) -> list[Job]:
    if phase not in PRODUCTION_PHASES:
        raise ValueError(f"unknown phase: {phase}")
    if setting not in protocol["settings"]:
        raise ValueError(f"unknown setting: {setting}")
    invalid_horizons = sorted(set(horizons) - set(HORIZONS))
    if invalid_horizons:
        raise ValueError(f"horizons outside frozen protocol: {invalid_horizons}")

    setting_spec = protocol["settings"][setting]
    candidate_count = len(
        protocol["candidate_grids"][setting_spec["optimizer_key"]]["candidates"]
    )
    jobs: list[Job] = []
    if phase == "arm-a":
        if setting != "S1" or replicate_mode != "pilot":
            raise ValueError("Arm A is frozen to S1 and replicate mode pilot")
        for scheduler in CORE_SCHEDULERS:
            for candidate_index in range(candidate_count):
                for budget in horizons:
                    for seed_slot in range(3):
                        jobs.append(
                            Job(
                                phase,
                                setting,
                                scheduler,
                                0,
                                candidate_index,
                                budget,
                                seed_slot,
                            )
                        )
        return jobs

    for scheduler in CORE_SCHEDULERS:
        for replicate in scheduler_replicates(scheduler, replicate_mode):
            if phase == "hpo":
                jobs.extend(
                    Job(
                        phase,
                        setting,
                        scheduler,
                        replicate,
                        candidate_index,
                        TUNE_BUDGET,
                        0,
                    )
                    for candidate_index in range(candidate_count)
                )
            else:
                candidate_index = select_candidate(
                    output_root, protocol, setting, scheduler, replicate
                )
                jobs.extend(
                    Job(
                        phase,
                        setting,
                        scheduler,
                        replicate,
                        candidate_index,
                        budget,
                        seed_slot,
                    )
                    for budget in horizons
                    for seed_slot in range(3)
                )
    return jobs


def job_seed(protocol: dict[str, Any], job: Job) -> int:
    record = protocol["seeds"]["settings"][job.setting][job.hpo_replicate]
    if job.stage == "hpo":
        return int(record["hpo_train_seed"])
    return int(record["deployment_seeds"][job.seed_slot])


def job_run_id(protocol: dict[str, Any], job: Job) -> str:
    return make_run_id(
        protocol["protocol_hash"],
        job.stage,
        job.setting,
        job.scheduler,
        job.hpo_replicate,
        job.candidate_index,
        job.budget,
        job_seed(protocol, job),
    )


def job_output(output_root: Path, protocol: dict[str, Any], job: Job) -> Path:
    return (
        output_root
        / job.stage
        / job.setting
        / job.scheduler
        / f"{job_run_id(protocol, job)}.parquet"
    )


def runner_command(
    job: Job,
    protocol_path: Path,
    data_manifest: Path,
    data_root: Path,
    output_root: Path,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("confirmatory_runner.py")),
        "--protocol",
        str(protocol_path),
        "--data-manifest",
        str(data_manifest),
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
        "--stage",
        job.stage,
        "--setting",
        job.setting,
        "--scheduler",
        job.scheduler,
        "--hpo-replicate",
        str(job.hpo_replicate),
        "--candidate-index",
        str(job.candidate_index),
        "--budget",
        str(job.budget),
        "--seed-slot",
        str(job.seed_slot),
        "--device",
        device,
    ]


def lane_jobs(jobs: list[Job], lane_index: int, lane_count: int) -> list[Job]:
    if lane_count < 1 or not 0 <= lane_index < lane_count:
        raise ValueError("lane index must lie in [0, lane count)")
    return [job for index, job in enumerate(jobs) if index % lane_count == lane_index]


def execute_jobs(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    jobs = build_jobs(
        protocol,
        args.output_root,
        args.phase,
        args.setting,
        args.replicates,
        args.horizons,
    )
    assigned = lane_jobs(jobs, args.lane_index, args.lane_count)
    if args.max_jobs is not None:
        assigned = assigned[: args.max_jobs]
    completed = 0
    for index, job in enumerate(assigned, start=1):
        command = runner_command(
            job,
            args.protocol,
            args.data_manifest,
            args.data_root,
            args.output_root,
            args.device,
        )
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)
        completed += 1
        print(
            f"lane {args.lane_index}/{args.lane_count} completed "
            f"{index}/{len(assigned)} {job_run_id(protocol, job)}",
            flush=True,
        )
    print(
        f"phase complete: phase={args.phase} setting={args.setting} "
        f"lane={args.lane_index}/{args.lane_count} jobs={completed}",
        flush=True,
    )


def recompute_metric_hash(frame: pl.DataFrame) -> str:
    return content_hash(
        {
            "val_score": [
                metric_token(value) for value in frame["val_score"].to_list()
            ],
            "test_score": [
                metric_token(value) for value in frame["test_score"].to_list()
            ],
        }
    )


def expected_scheduler_params(
    protocol: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    shape = protocol["scheduler_shapes"][summary["scheduler"]]
    return {
        "peak_lr": summary["lr0"],
        "infimum_lr": summary["lr0"] * shape.get("infimum_ratio", 0.0),
        "max_iter": summary["budget"],
        **shape,
    }


def validate_artifact(
    output_root: Path,
    protocol: dict[str, Any],
    data_manifest: dict[str, Any],
    job: Job,
) -> dict[str, Any]:
    curve_path = job_output(output_root, protocol, job)
    summary_path = curve_path.with_suffix(".summary.json")
    if not curve_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"missing artifact pair for {job_run_id(protocol, job)}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frame = pl.read_parquet(curve_path)
    run_id = job_run_id(protocol, job)
    expected_identity = {
        "run_id": run_id,
        "stage": job.stage,
        "setting": job.setting,
        "scheduler": job.scheduler,
        "hpo_replicate": job.hpo_replicate,
        "candidate_index": job.candidate_index,
        "budget": job.budget,
    }
    for key, expected in expected_identity.items():
        if summary.get(key) != expected:
            raise RuntimeError(
                f"artifact identity mismatch for {run_id}.{key}: "
                f"{summary.get(key)} != {expected}"
            )
    expected_selection = make_selection_id(
        protocol["protocol_hash"], job.setting, job.scheduler, job.hpo_replicate
    )
    if summary["selection_id"] != expected_selection:
        raise RuntimeError(f"selection identity mismatch for {run_id}")
    if set(OUTPUT_SCHEMA) - set(frame.columns):
        missing = sorted(set(OUTPUT_SCHEMA) - set(frame.columns))
        raise RuntimeError(f"missing output fields for {run_id}: {missing}")
    metadata_columns = OUTPUT_SCHEMA[: OUTPUT_SCHEMA.index("epoch")]
    drifting = [column for column in metadata_columns if frame[column].n_unique() != 1]
    if drifting:
        raise RuntimeError(f"metadata drift within {run_id}: {drifting}")
    if frame.height != job.budget or int(summary["epochs"]) != job.budget:
        raise RuntimeError(f"epoch count mismatch for {run_id}")
    for key in metadata_columns:
        if key in summary and frame[key][0] != summary[key]:
            raise RuntimeError(f"curve/summary metadata mismatch for {run_id}.{key}")
    if summary["protocol_hash"] != protocol["protocol_hash"]:
        raise RuntimeError(f"protocol hash mismatch for {run_id}")
    if summary["split_hash"] != data_manifest["split_hash"]:
        raise RuntimeError(f"split hash mismatch for {run_id}")
    setting = protocol["settings"][job.setting]
    if data_manifest["dataset_group"] != setting["dataset_group"]:
        raise RuntimeError(f"dataset group mismatch for {run_id}")
    grid_hash = protocol["candidate_grids"][setting["optimizer_key"]]["sha256"]
    if summary["candidate_grid_hash"] != grid_hash:
        raise RuntimeError(f"candidate grid hash mismatch for {run_id}")
    if summary["metric_direction"] != TASK_SCORE_DIRECTIONS[setting["task"]]:
        raise RuntimeError(f"metric direction mismatch for {run_id}")
    if json.loads(summary["scheduler_params"]) != expected_scheduler_params(
        protocol, summary
    ):
        raise RuntimeError(f"scheduler parameter drift for {run_id}")
    if file_sha256(curve_path) != summary["curve_sha256"]:
        raise RuntimeError(f"curve checksum mismatch for {run_id}")
    if recompute_metric_hash(frame) != summary["metric_hash"]:
        raise RuntimeError(f"metric hash mismatch for {run_id}")
    if summary["dirty"] or summary["run_status"] != "complete":
        raise RuntimeError(f"unfrozen production provenance for {run_id}")
    should_access_test = job.stage != "hpo"
    if bool(summary["test_accessed"]) != should_access_test:
        raise RuntimeError(f"test-access contract mismatch for {run_id}")
    if not should_access_test:
        if summary["test_metric"] is not None:
            raise RuntimeError(f"HPO summary contains test metric for {run_id}")
        if frame["test_score"].null_count() != frame.height:
            raise RuntimeError(f"HPO curve contains test scores for {run_id}")
    elif summary["test_metric"] is None:
        raise RuntimeError(f"deployment summary lacks test metric for {run_id}")
    return summary


def audit_jobs(
    output_root: Path,
    protocol: dict[str, Any],
    data_manifest_path: Path,
    phases: tuple[str, ...],
    setting: str,
    replicate_mode: str,
    horizons: tuple[int, ...],
    require_interior_selection: bool,
) -> dict[str, Any]:
    data_manifest = load_data_manifest(data_manifest_path, protocol["protocol_hash"])
    expected_jobs = [
        job
        for phase in phases
        for job in build_jobs(
            protocol, output_root, phase, setting, replicate_mode, horizons
        )
    ]
    expected_ids = {job_run_id(protocol, job) for job in expected_jobs}
    actual_summary_paths = [
        path
        for phase in phases
        for path in (output_root / phase / setting).glob("*/*.summary.json")
    ]
    actual_summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in actual_summary_paths
    ]
    actual_ids = {summary["run_id"] for summary in actual_summaries}
    actual_curve_paths = [
        path
        for phase in phases
        for path in (output_root / phase / setting).glob("*/*.parquet")
    ]
    actual_curve_ids = {path.stem for path in actual_curve_paths}
    if actual_curve_ids != expected_ids or len(actual_curve_ids) != len(
        actual_curve_paths
    ):
        missing = sorted(expected_ids - actual_curve_ids)
        extra = sorted(actual_curve_ids - expected_ids)
        raise RuntimeError(
            f"production curve key mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise RuntimeError(
            f"production run key mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    if len(actual_ids) != len(actual_summaries):
        raise RuntimeError("duplicate production run IDs")
    partials = sorted(output_root.rglob("*.partial"))
    if partials:
        raise RuntimeError(f"partial production artifacts remain: {partials[:5]}")

    summaries = [
        validate_artifact(output_root, protocol, data_manifest, job)
        for job in expected_jobs
    ]
    commits = sorted({summary["code_commit"] for summary in summaries})
    environment_hashes = sorted(
        {summary["environment_lock_hash"] for summary in summaries}
    )
    if len(commits) != 1 or len(environment_hashes) != 1:
        raise RuntimeError(
            f"mixed provenance: commits={commits}, environment_hashes={environment_hashes}"
        )

    selected: dict[str, int] = {}
    boundary: list[str] = []
    if "hpo" in phases or "deploy" in phases:
        setting_spec = protocol["settings"][setting]
        candidate_count = len(
            protocol["candidate_grids"][setting_spec["optimizer_key"]]["candidates"]
        )
        for scheduler in CORE_SCHEDULERS:
            for replicate in scheduler_replicates(scheduler, replicate_mode):
                candidate = select_candidate(
                    output_root, protocol, setting, scheduler, replicate
                )
                key = f"{scheduler}:R{replicate}"
                selected[key] = candidate
                if candidate in {0, candidate_count - 1}:
                    boundary.append(key)

    hpo_seconds = [
        float(summary["elapsed_seconds"])
        for summary in summaries
        if summary["stage"] == "hpo"
    ]
    deploy_seconds_per_epoch = [
        float(summary["elapsed_seconds"]) / int(summary["epochs"])
        for summary in summaries
        if summary["stage"] in {"deploy", "arm-a"}
    ]
    report = {
        "protocol_hash": protocol["protocol_hash"],
        "split_hash": data_manifest["split_hash"],
        "setting": setting,
        "replicate_mode": replicate_mode,
        "phases": list(phases),
        "horizons": list(horizons),
        "expected_key_count": len(expected_ids),
        "artifact_count": len(summaries),
        "code_commit": commits[0],
        "environment_lock_hash": environment_hashes[0],
        "selected_candidates": selected,
        "boundary_selections": boundary,
        "nonfinite_count": sum(
            summary["method_outcome"] == "nonfinite" for summary in summaries
        ),
        "median_seconds_per_hpo_trial": (
            statistics.median(hpo_seconds) if hpo_seconds else None
        ),
        "median_seconds_per_deployment_epoch": (
            statistics.median(deploy_seconds_per_epoch)
            if deploy_seconds_per_epoch
            else None
        ),
    }
    report["audit_hash"] = content_hash(report)
    report_name = (
        f"audit_{setting}_{replicate_mode}_{'-'.join(phases)}_"
        f"{'-'.join(str(value) for value in horizons)}.json"
    )
    report_path = output_root / report_name
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if require_interior_selection and boundary:
        raise RuntimeError(f"search-boundary gate failed: {boundary}")
    return report


def print_plan(jobs: Iterable[Job], protocol: dict[str, Any]) -> None:
    rows = [{**asdict(job), "run_id": job_run_id(protocol, job)} for job in jobs]
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['stage']}:{row['scheduler']}"
        counts[key] = counts.get(key, 0) + 1
    print(
        json.dumps(
            {
                "job_count": len(rows),
                "counts": dict(sorted(counts.items())),
                "plan_hash": content_hash(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


def add_scope_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--setting", choices=("S1", "S2", "S3", "S4", "S5"), required=True
    )
    parser.add_argument("--replicates", choices=("pilot", "full"), required=True)
    parser.add_argument(
        "--horizons",
        type=parse_csv_ints,
        default=HORIZONS,
        help="comma-separated deployment horizons",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    add_scope_arguments(plan_parser)
    plan_parser.add_argument("--phase", choices=PRODUCTION_PHASES, required=True)

    run_parser = subparsers.add_parser("run")
    add_scope_arguments(run_parser)
    run_parser.add_argument("--phase", choices=PRODUCTION_PHASES, required=True)
    run_parser.add_argument("--data-manifest", type=Path, required=True)
    run_parser.add_argument("--data-root", type=Path, required=True)
    run_parser.add_argument("--device", required=True)
    run_parser.add_argument("--lane-index", type=int, required=True)
    run_parser.add_argument("--lane-count", type=int, required=True)
    run_parser.add_argument("--max-jobs", type=int)

    audit_parser = subparsers.add_parser("audit")
    add_scope_arguments(audit_parser)
    audit_parser.add_argument("--data-manifest", type=Path, required=True)
    audit_parser.add_argument("--phases", type=parse_csv_phases, required=True)
    audit_parser.add_argument("--require-interior-selection", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_manifest(args.protocol)
    if args.command == "plan":
        jobs = build_jobs(
            protocol,
            args.output_root,
            args.phase,
            args.setting,
            args.replicates,
            args.horizons,
        )
        print_plan(jobs, protocol)
    elif args.command == "run":
        execute_jobs(args, protocol)
    else:
        report = audit_jobs(
            args.output_root,
            protocol,
            args.data_manifest,
            args.phases,
            args.setting,
            args.replicates,
            args.horizons,
            args.require_interior_selection,
        )
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
