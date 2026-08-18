"""Plan, execute, and audit the v13 deterministic G1a dry campaign."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import polars as pl

from confirmatory_data import load_data_manifest
from confirmatory_protocol import (
    CORE_SCHEDULERS,
    OUTPUT_SCHEMA,
    TASK_SCORE_DIRECTIONS,
    TUNE_BUDGET,
    content_hash,
    file_sha256,
    load_manifest,
    make_run_id,
    metric_token,
)
from confirmatory_runner import run

DRY_HORIZONS = (10, 200)
DRY_SETTING = "S1"


def write_dry_data_manifest(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "protocol_hash": protocol["protocol_hash"],
        "dataset_group": "dry-v13",
        "kind": "synthetic-regression",
        "generator": "confirmatory_runner.dry_data",
        "splits": {
            "train": {"count": 32},
            "validation": {"count": 8},
            "test": {"count": 8},
        },
    }
    manifest["split_hash"] = content_hash(manifest)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise RuntimeError(
            f"existing dry data manifest differs from frozen content: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return manifest


def runner_args(
    protocol_path: Path,
    data_manifest_path: Path,
    output_root: Path,
    stage: str,
    scheduler: str,
    candidate_index: int,
    budget: int,
    seed_slot: int,
    device: str,
) -> Namespace:
    return Namespace(
        protocol=protocol_path,
        data_manifest=data_manifest_path,
        data_root=Path("unused-for-dry"),
        output_root=output_root,
        stage=stage,
        setting=DRY_SETTING,
        scheduler=scheduler,
        hpo_replicate=0,
        candidate_index=candidate_index,
        budget=budget,
        seed_slot=seed_slot,
        device=device,
        allow_dirty=True,
    )


def call_runner(args: Namespace, verbose: bool) -> tuple[Path, bool]:
    before = set(args.output_root.rglob("*.parquet"))
    if verbose:
        path = run(args)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            path = run(args)
    return path, path not in before


def read_hpo_summaries(output_root: Path, scheduler: str) -> list[dict[str, Any]]:
    root = output_root / "dry-hpo" / DRY_SETTING / scheduler
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in root.glob("*.summary.json")
    ]
    return sorted(summaries, key=lambda row: row["candidate_index"])


def selected_candidate(
    output_root: Path,
    scheduler: str,
    expected_count: int,
    task: str,
) -> int:
    summaries = read_hpo_summaries(output_root, scheduler)
    if len(summaries) != expected_count:
        raise RuntimeError(
            f"cannot select {scheduler}: {len(summaries)} of {expected_count} candidates"
        )
    finite = [
        row for row in summaries if isinstance(row["validation_metric"], (int, float))
    ]
    if not finite:
        raise RuntimeError(f"all dry HPO candidates are non-finite for {scheduler}")
    direction = TASK_SCORE_DIRECTIONS[task]
    sign = 1.0 if direction == "minimize" else -1.0
    best = min(
        finite,
        key=lambda row: (
            sign * row["validation_metric"],
            row["candidate_index"],
        ),
    )
    return int(best["candidate_index"])


def expected_dry_run_ids(
    protocol: dict[str, Any],
    selected: dict[str, int],
) -> set[str]:
    protocol_hash = protocol["protocol_hash"]
    seeds = protocol["seeds"]["settings"][DRY_SETTING][0]
    candidate_count = len(protocol["candidate_grids"]["adamw"]["candidates"])
    expected = {
        make_run_id(
            protocol_hash,
            "dry-hpo",
            DRY_SETTING,
            scheduler,
            0,
            candidate_index,
            TUNE_BUDGET,
            seeds["hpo_train_seed"],
        )
        for scheduler in CORE_SCHEDULERS
        for candidate_index in range(candidate_count)
    }
    expected.update(
        make_run_id(
            protocol_hash,
            "dry-deploy",
            DRY_SETTING,
            scheduler,
            0,
            selected[scheduler],
            budget,
            deployment_seed,
        )
        for scheduler in CORE_SCHEDULERS
        for budget in DRY_HORIZONS
        for deployment_seed in seeds["deployment_seeds"]
    )
    return expected


def run_dry(args: argparse.Namespace) -> None:
    protocol = load_manifest(args.protocol)
    data_manifest_path = args.output_root / "dry_data_manifest.json"
    write_dry_data_manifest(data_manifest_path, protocol)
    candidate_count = len(protocol["candidate_grids"]["adamw"]["candidates"])
    calls = 0
    executed = 0
    skipped = 0

    def launch(job: Namespace) -> bool:
        nonlocal calls, executed, skipped
        if args.max_jobs is not None and calls >= args.max_jobs:
            return False
        _, was_executed = call_runner(job, args.verbose)
        calls += 1
        executed += int(was_executed)
        skipped += int(not was_executed)
        return True

    for scheduler in CORE_SCHEDULERS:
        for candidate_index in range(candidate_count):
            job = runner_args(
                args.protocol,
                data_manifest_path,
                args.output_root,
                "dry-hpo",
                scheduler,
                candidate_index,
                TUNE_BUDGET,
                0,
                args.device,
            )
            if not launch(job):
                print(
                    f"partial dry campaign: calls={calls} executed={executed} skipped={skipped}"
                )
                return

    selected = {
        scheduler: selected_candidate(
            args.output_root,
            scheduler,
            candidate_count,
            protocol["settings"][DRY_SETTING]["task"],
        )
        for scheduler in CORE_SCHEDULERS
    }
    for scheduler in CORE_SCHEDULERS:
        for budget in DRY_HORIZONS:
            for seed_slot in range(3):
                job = runner_args(
                    args.protocol,
                    data_manifest_path,
                    args.output_root,
                    "dry-deploy",
                    scheduler,
                    selected[scheduler],
                    budget,
                    seed_slot,
                    args.device,
                )
                if not launch(job):
                    print(
                        f"partial dry campaign: calls={calls} executed={executed} skipped={skipped}"
                    )
                    return

    report = audit_dry(args.output_root, protocol, data_manifest_path)
    print(
        f"dry campaign complete: executed={executed} skipped={skipped} "
        f"curves={report['curve_count']} audit_hash={report['audit_hash']}"
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


def audit_dry(
    output_root: Path,
    protocol: dict[str, Any],
    data_manifest_path: Path | None = None,
) -> dict[str, Any]:
    data_manifest_path = data_manifest_path or output_root / "dry_data_manifest.json"
    data_manifest = load_data_manifest(data_manifest_path, protocol["protocol_hash"])
    candidate_count = len(protocol["candidate_grids"]["adamw"]["candidates"])
    expected_hpo = len(CORE_SCHEDULERS) * candidate_count
    expected_deploy = len(CORE_SCHEDULERS) * len(DRY_HORIZONS) * 3
    summary_paths = sorted(output_root.glob("dry-*/*/*/*.summary.json"))
    curve_paths = sorted(output_root.glob("dry-*/*/*/*.parquet"))
    partial_paths = sorted(output_root.rglob("*.partial"))
    if partial_paths:
        raise RuntimeError(f"partial artifacts remain: {partial_paths[:3]}")
    if len(summary_paths) != expected_hpo + expected_deploy:
        raise RuntimeError(
            f"summary count mismatch: {len(summary_paths)} != {expected_hpo + expected_deploy}"
        )
    if len(curve_paths) != len(summary_paths):
        raise RuntimeError(
            f"curve/summary mismatch: {len(curve_paths)} != {len(summary_paths)}"
        )

    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    run_ids = [summary["run_id"] for summary in summaries]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("duplicate run_id in dry campaign")
    hpo_count = 0
    deploy_count = 0
    selected = {
        scheduler: selected_candidate(
            output_root,
            scheduler,
            candidate_count,
            protocol["settings"][DRY_SETTING]["task"],
        )
        for scheduler in CORE_SCHEDULERS
    }
    expected_run_ids = expected_dry_run_ids(protocol, selected)
    actual_run_ids = set(run_ids)
    if actual_run_ids != expected_run_ids:
        missing = sorted(expected_run_ids - actual_run_ids)
        extra = sorted(actual_run_ids - expected_run_ids)
        raise RuntimeError(
            f"dry run key mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    metric_hashes: dict[str, str] = {}

    by_run = {summary["run_id"]: summary for summary in summaries}
    for curve_path in curve_paths:
        frame = pl.read_parquet(curve_path)
        run_id = frame["run_id"][0]
        summary = by_run.get(run_id)
        if summary is None:
            raise RuntimeError(f"curve has no matching summary: {curve_path}")
        if not set(OUTPUT_SCHEMA).issubset(frame.columns):
            missing = sorted(set(OUTPUT_SCHEMA) - set(frame.columns))
            raise RuntimeError(f"missing output fields in {curve_path}: {missing}")
        metadata_columns = OUTPUT_SCHEMA[: OUTPUT_SCHEMA.index("epoch")]
        drifting = [
            column for column in metadata_columns if frame[column].n_unique() != 1
        ]
        if drifting:
            raise RuntimeError(f"metadata drift within {curve_path}: {drifting}")
        if frame.height != int(summary["budget"]):
            raise RuntimeError(
                f"epoch count mismatch in {curve_path}: {frame.height} != {summary['budget']}"
            )
        if frame["protocol_hash"][0] != protocol["protocol_hash"]:
            raise RuntimeError(f"protocol hash mismatch in {curve_path}")
        if frame["split_hash"][0] != data_manifest["split_hash"]:
            raise RuntimeError(f"split hash mismatch in {curve_path}")
        setting = protocol["settings"][summary["setting"]]
        expected_grid_hash = protocol["candidate_grids"][setting["optimizer_key"]][
            "sha256"
        ]
        if summary["candidate_grid_hash"] != expected_grid_hash:
            raise RuntimeError(f"candidate grid hash mismatch in {curve_path}")
        if summary["metric_direction"] != TASK_SCORE_DIRECTIONS[setting["task"]]:
            raise RuntimeError(f"metric direction mismatch in {curve_path}")
        if file_sha256(curve_path) != summary["curve_sha256"]:
            raise RuntimeError(f"curve checksum mismatch: {curve_path}")
        metric_hash = recompute_metric_hash(frame)
        if metric_hash != summary["metric_hash"]:
            raise RuntimeError(f"metric hash mismatch: {curve_path}")
        metric_hashes[run_id] = metric_hash
        scheduler = summary["scheduler"]
        actual_params = json.loads(summary["scheduler_params"])
        expected_params = {
            "peak_lr": summary["lr0"],
            "infimum_lr": summary["lr0"]
            * protocol["scheduler_shapes"][scheduler].get("infimum_ratio", 0.0),
            "max_iter": summary["budget"],
            **protocol["scheduler_shapes"][scheduler],
        }
        if actual_params != expected_params:
            raise RuntimeError(
                f"scheduler parameter drift for {scheduler}: {actual_params} != {expected_params}"
            )
        if summary["stage"] == "dry-hpo":
            hpo_count += 1
            if summary["test_accessed"] or summary["test_metric"] is not None:
                raise RuntimeError(f"HPO accessed test data: {run_id}")
            if frame["test_score"].null_count() != frame.height:
                raise RuntimeError(f"HPO curve contains test scores: {run_id}")
        elif summary["stage"] == "dry-deploy":
            deploy_count += 1
            if not summary["test_accessed"] or summary["test_metric"] is None:
                raise RuntimeError(f"deployment did not evaluate test data: {run_id}")
            if summary["candidate_index"] != selected[scheduler]:
                raise RuntimeError(
                    f"deployment did not use selected candidate: {run_id}"
                )
        else:
            raise RuntimeError(f"unexpected dry stage: {summary['stage']}")

    if hpo_count != expected_hpo or deploy_count != expected_deploy:
        raise RuntimeError(
            f"stage counts mismatch: hpo={hpo_count}/{expected_hpo}, deploy={deploy_count}/{expected_deploy}"
        )
    report = {
        "protocol_hash": protocol["protocol_hash"],
        "split_hash": data_manifest["split_hash"],
        "dry_manifest_sha256": file_sha256(data_manifest_path),
        "curve_count": len(curve_paths),
        "hpo_count": hpo_count,
        "deploy_count": deploy_count,
        "selected_candidates": selected,
        "expected_key_count": len(expected_run_ids),
        "metric_hashes": dict(sorted(metric_hashes.items())),
    }
    report["audit_hash"] = content_hash(report)
    report_path = output_root / "dry_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def compare_dry(left: Path, right: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    left_report = audit_dry(left, protocol)
    right_report = audit_dry(right, protocol)
    checks = {
        "protocol_hash_equal": left_report["protocol_hash"]
        == right_report["protocol_hash"],
        "dry_manifest_bytes_equal": left_report["dry_manifest_sha256"]
        == right_report["dry_manifest_sha256"],
        "selected_candidates_equal": left_report["selected_candidates"]
        == right_report["selected_candidates"],
        "metric_hashes_equal": left_report["metric_hashes"]
        == right_report["metric_hashes"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"dry replay mismatch: {checks}")
    report = {
        **checks,
        "left_audit_hash": left_report["audit_hash"],
        "right_audit_hash": right_report["audit_hash"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-dry")
    run_parser.add_argument("--protocol", type=Path, required=True)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--device", default="cpu")
    run_parser.add_argument("--max-jobs", type=int)
    run_parser.add_argument("--verbose", action="store_true")

    audit_parser = subparsers.add_parser("audit-dry")
    audit_parser.add_argument("--protocol", type=Path, required=True)
    audit_parser.add_argument("--output-root", type=Path, required=True)

    compare_parser = subparsers.add_parser("compare-dry")
    compare_parser.add_argument("--protocol", type=Path, required=True)
    compare_parser.add_argument("--left", type=Path, required=True)
    compare_parser.add_argument("--right", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_manifest(args.protocol)
    if args.command == "run-dry":
        run_dry(args)
    elif args.command == "audit-dry":
        report = audit_dry(args.output_root, protocol)
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "metric_hashes"},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        compare_dry(args.left, args.right, protocol)


if __name__ == "__main__":
    main()
