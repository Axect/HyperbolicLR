"""Frozen protocol and provenance helpers for the v13 confirmatory campaign.

The scientific protocol lives here as executable data. Runners may read this module or a
manifest produced by it, but they must not redefine candidate grids, seeds, scheduler shapes,
or output fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

PROTOCOL_NAME = "hyperboliclr-v13-confirmatory"
PROTOCOL_VERSION = "2026-08-18.2"
MASTER_SEED = 20260818
TUNE_BUDGET = 10
HORIZONS = (10, 50, 100, 200)
DEPLOYMENT_SEEDS_PER_SELECTION = 3
CONFIRMATORY_REPLICATES = 12
DESCRIPTIVE_REPLICATES = 3
FINAL_WINDOW_FRACTION = 0.1
FINAL_WINDOW_MIN = 3

CORE_SCHEDULERS = ("EH", "H", "E", "C", "D2Z", "WSD", "POW", "WSQD")
CONFIRMATORY_SCHEDULERS = ("EH", "H", "POW", "WSQD")
DESCRIPTIVE_SCHEDULERS = ("E", "C", "D2Z", "WSD")
TASK_SCORE_DIRECTIONS = {"integral": "minimize", "cifar10": "maximize"}

GRID_SPECS = {
    "adamw": {"lower": 1e-5, "upper": 5e-2, "count": 30},
    "sgd": {"lower": 1e-3, "upper": 1.0, "count": 30},
}

SCHEDULER_SHAPES: dict[str, dict[str, Any]] = {
    "EH": {"upper_bound": 400, "infimum_ratio": 1e-3},
    "H": {"upper_bound": 400, "infimum_ratio": 1e-3},
    "E": {"gamma": 0.963},
    "C": {"infimum_ratio": 1e-3},
    "D2Z": {},
    "WSD": {"cooldown_frac": 0.2, "infimum_ratio": 1e-3, "warmup_fraction": 0.1},
    "POW": {"pow_b": -0.51, "pow_decay_frac": 0.1, "warmup_epochs": 5},
    "WSQD": {"decay_frac": 0.2, "shift": 10.0, "warmup_epochs": 5},
}

SETTINGS: dict[str, dict[str, Any]] = {
    "S1": {
        "task": "integral",
        "dataset_group": "operator-grf-v13",
        "model": "DeepONet",
        "optimizer": "AdamW",
        "optimizer_key": "adamw",
        "optimizer_params": {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
        "batch_size": 100,
        "required": True,
    },
    "S2": {
        "task": "integral",
        "dataset_group": "operator-grf-v13",
        "model": "TFONet",
        "optimizer": "AdamW",
        "optimizer_key": "adamw",
        "optimizer_params": {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
        "batch_size": 100,
        "required": True,
    },
    "S3": {
        "task": "cifar10",
        "dataset_group": "cifar10-v13",
        "model": "ResNetCIFAR18",
        "optimizer": "SGD",
        "optimizer_key": "sgd",
        "optimizer_params": {"momentum": 0.9, "weight_decay": 5e-4},
        "batch_size": 128,
        "required": True,
    },
    "S4": {
        "task": "integral",
        "dataset_group": "operator-grf-v13",
        "model": "DeepONetLN",
        "optimizer": "AdamW",
        "optimizer_key": "adamw",
        "optimizer_params": {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
        "batch_size": 100,
        "required": False,
    },
    "S5": {
        "task": "cifar10",
        "dataset_group": "cifar10-v13",
        "model": "SimpleViT",
        "optimizer": "AdamW",
        "optimizer_key": "adamw",
        "optimizer_params": {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
        "batch_size": 128,
        "required": False,
    },
}

OUTPUT_SCHEMA = (
    "run_id",
    "selection_id",
    "protocol_hash",
    "stage",
    "setting",
    "task",
    "model",
    "scheduler",
    "hpo_replicate",
    "candidate_index",
    "hpo_train_seed",
    "deployment_seed",
    "split_hash",
    "candidate_grid_hash",
    "optimizer",
    "optimizer_params",
    "scheduler_params",
    "tune_budget",
    "budget",
    "lr0",
    "code_commit",
    "dirty",
    "environment_lock_hash",
    "device",
    "run_status",
    "method_outcome",
    "test_accessed",
    "metric_direction",
    "epoch",
    "train_loss",
    "val_loss",
    "val_score",
    "test_loss",
    "test_score",
    "lr",
)


def canonical_json(value: Any) -> str:
    """Return the one canonical JSON representation used by every protocol hash."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def metric_token(value: float | None) -> float | str | None:
    """Represent non-finite metrics in canonical JSON without invalid NaN/Infinity tokens."""
    if value is None:
        return None
    value = float(value)
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log_grid(spec: dict[str, Any]) -> list[float]:
    lower = float(spec["lower"])
    upper = float(spec["upper"])
    count = int(spec["count"])
    if not 0 < lower < upper or count < 2:
        raise ValueError(f"invalid grid spec: {spec}")
    ratio = upper / lower
    return [lower * ratio ** (index / (count - 1)) for index in range(count)]


def candidate_grid(optimizer_key: str) -> list[dict[str, Any]]:
    if optimizer_key not in GRID_SPECS:
        raise KeyError(f"unknown optimizer grid: {optimizer_key}")
    return [
        {"candidate_index": index, "lr": lr}
        for index, lr in enumerate(log_grid(GRID_SPECS[optimizer_key]))
    ]


def candidate_grid_hash(optimizer_key: str) -> str:
    return content_hash(candidate_grid(optimizer_key))


def derive_seed(role: str, group: str, replicate: int = 0, slot: str | int = 0) -> int:
    key = f"{PROTOCOL_NAME}|{PROTOCOL_VERSION}|{MASTER_SEED}|{role}|{group}|{replicate}|{slot}"
    raw = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return raw % (2**31 - 2) + 1


def replicate_count(scheduler: str) -> int:
    if scheduler in CONFIRMATORY_SCHEDULERS:
        return CONFIRMATORY_REPLICATES
    if scheduler in DESCRIPTIVE_SCHEDULERS:
        return DESCRIPTIVE_REPLICATES
    raise KeyError(f"scheduler is outside the frozen core set: {scheduler}")


def final_window(budget: int) -> int:
    if budget < 1:
        raise ValueError("budget must be positive")
    return max(FINAL_WINDOW_MIN, math.ceil(FINAL_WINDOW_FRACTION * budget))


def seed_manifest() -> dict[str, Any]:
    dataset_groups = sorted({setting["dataset_group"] for setting in SETTINGS.values()})
    datasets = {
        group: {
            "train": derive_seed("dataset", group, slot="train"),
            "validation": derive_seed("dataset", group, slot="validation"),
            "test": derive_seed("dataset", group, slot="test"),
            "split": derive_seed("split", group),
        }
        for group in dataset_groups
    }
    settings: dict[str, Any] = {}
    for setting_id in SETTINGS:
        replicates = []
        for replicate in range(CONFIRMATORY_REPLICATES):
            replicates.append(
                {
                    "hpo_replicate": replicate,
                    "hpo_train_seed": derive_seed("hpo-train", setting_id, replicate),
                    "deployment_seeds": [
                        derive_seed("deployment", setting_id, replicate, slot)
                        for slot in range(DEPLOYMENT_SEEDS_PER_SELECTION)
                    ],
                }
            )
        settings[setting_id] = replicates
    return {"datasets": datasets, "settings": settings}


def build_manifest() -> dict[str, Any]:
    grids = {
        optimizer: {
            "spec": spec,
            "candidates": candidate_grid(optimizer),
            "sha256": candidate_grid_hash(optimizer),
        }
        for optimizer, spec in GRID_SPECS.items()
    }
    manifest: dict[str, Any] = {
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "master_seed": MASTER_SEED,
        "tune_budget": TUNE_BUDGET,
        "horizons": list(HORIZONS),
        "deployment_seeds_per_selection": DEPLOYMENT_SEEDS_PER_SELECTION,
        "confirmatory_replicates": CONFIRMATORY_REPLICATES,
        "descriptive_replicates": DESCRIPTIVE_REPLICATES,
        "core_schedulers": list(CORE_SCHEDULERS),
        "confirmatory_schedulers": list(CONFIRMATORY_SCHEDULERS),
        "descriptive_schedulers": list(DESCRIPTIVE_SCHEDULERS),
        "scheduler_shapes": SCHEDULER_SHAPES,
        "settings": SETTINGS,
        "candidate_grids": grids,
        "seeds": seed_manifest(),
        "metrics": {
            "final_window_fraction": FINAL_WINDOW_FRACTION,
            "final_window_min": FINAL_WINDOW_MIN,
            "hpo_split": "validation",
            "confirmatory_split": "test",
            "regression_score": "median final-window loss",
            "regression_direction": TASK_SCORE_DIRECTIONS["integral"],
            "classification_score": "median final-window accuracy",
            "selection_tie_break": "lowest candidate_index",
            "classification_direction": TASK_SCORE_DIRECTIONS["cifar10"],
        },
        "primary_comparisons": {
            "EH_vs_H": {"kind": "superiority", "margin": 0.0},
            "EH_vs_POW_regression": {
                "kind": "noninferiority",
                "margin_log10_loss": -0.1,
            },
            "EH_vs_WSQD_regression": {
                "kind": "noninferiority",
                "margin_log10_loss": -0.1,
            },
            "EH_vs_POW_classification": {
                "kind": "noninferiority",
                "margin_accuracy_points": -0.5,
            },
            "EH_vs_WSQD_classification": {
                "kind": "noninferiority",
                "margin_accuracy_points": -0.5,
            },
            "multiplicity": "Holm within each setting",
            "test": "one-sided exact paired sign-flip test",
        },
        "output_schema": list(OUTPUT_SCHEMA),
    }
    manifest["protocol_hash"] = content_hash(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    claimed_hash = manifest.get("protocol_hash")
    unhashed = {key: value for key, value in manifest.items() if key != "protocol_hash"}
    actual_hash = content_hash(unhashed)
    if claimed_hash != actual_hash:
        raise ValueError(
            f"protocol hash mismatch: claimed={claimed_hash}, actual={actual_hash}"
        )

    if tuple(manifest["core_schedulers"]) != CORE_SCHEDULERS:
        raise ValueError("core scheduler order differs from the frozen protocol")
    if tuple(manifest["horizons"]) != HORIZONS:
        raise ValueError("deployment horizons differ from the frozen protocol")
    if tuple(manifest["output_schema"]) != OUTPUT_SCHEMA:
        raise ValueError("output schema differs from the frozen protocol")
    metrics = manifest["metrics"]
    if metrics["regression_direction"] != TASK_SCORE_DIRECTIONS["integral"]:
        raise ValueError("regression score direction differs from the frozen protocol")
    if metrics["classification_direction"] != TASK_SCORE_DIRECTIONS["cifar10"]:
        raise ValueError(
            "classification score direction differs from the frozen protocol"
        )

    for optimizer_key in GRID_SPECS:
        grid = manifest["candidate_grids"][optimizer_key]
        if grid["sha256"] != content_hash(grid["candidates"]):
            raise ValueError(f"candidate grid hash mismatch for {optimizer_key}")
        if len(grid["candidates"]) != 30:
            raise ValueError(f"candidate grid for {optimizer_key} must have 30 points")

    seeds: list[int] = []
    for dataset in manifest["seeds"]["datasets"].values():
        seeds.extend(dataset.values())
    for setting_replicates in manifest["seeds"]["settings"].values():
        for replicate in setting_replicates:
            seeds.append(replicate["hpo_train_seed"])
            seeds.extend(replicate["deployment_seeds"])
    if len(seeds) != len(set(seeds)):
        raise ValueError("seed collision across protocol roles")

    for scheduler in CORE_SCHEDULERS:
        if scheduler not in manifest["scheduler_shapes"]:
            raise ValueError(f"missing frozen shape for scheduler {scheduler}")


def write_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    manifest = build_manifest()
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(payload, encoding="utf-8")
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    expected = build_manifest()["protocol_hash"]
    if manifest["protocol_hash"] != expected:
        raise ValueError(
            f"manifest is valid but not this source revision: {manifest['protocol_hash']} != {expected}"
        )
    return manifest


def make_selection_id(
    protocol_hash: str, setting: str, scheduler: str, replicate: int
) -> str:
    value = {
        "protocol_hash": protocol_hash,
        "setting": setting,
        "scheduler": scheduler,
        "replicate": replicate,
    }
    return "sel-" + content_hash(value)[:20]


def make_run_id(
    protocol_hash: str,
    stage: str,
    setting: str,
    scheduler: str,
    replicate: int,
    candidate_index: int,
    budget: int,
    seed: int,
) -> str:
    value = {
        "protocol_hash": protocol_hash,
        "stage": stage,
        "setting": setting,
        "scheduler": scheduler,
        "replicate": replicate,
        "candidate_index": candidate_index,
        "budget": budget,
        "seed": seed,
    }
    return "run-" + content_hash(value)[:24]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, help="write the frozen manifest")
    parser.add_argument("--validate", type=Path, help="validate an existing manifest")
    args = parser.parse_args()

    if bool(args.write) == bool(args.validate):
        parser.error("choose exactly one of --write or --validate")
    if args.write:
        manifest = write_manifest(args.write)
        print(f"wrote {args.write} protocol_hash={manifest['protocol_hash']}")
    else:
        manifest = load_manifest(args.validate)
        print(f"valid {args.validate} protocol_hash={manifest['protocol_hash']}")


if __name__ == "__main__":
    main()
