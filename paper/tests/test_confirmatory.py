import itertools
import json
import math

import pytest
import torch

from confirmatory_analysis import exact_sign_flip_pvalue, holm_adjust
from confirmatory_campaign import selected_candidate
from confirmatory_data import validate_data_files
from confirmatory_orchestrator import build_jobs, lane_jobs
from confirmatory_protocol import (
    CORE_SCHEDULERS,
    GRID_SPECS,
    TASK_SCORE_DIRECTIONS,
    build_manifest,
    candidate_grid,
    content_hash,
    file_sha256,
    final_window,
    metric_token,
    validate_manifest,
)
from confirmatory_runner import scheduler_parameters
from scheduler_config import create_scheduler


def _lr_curve(scheduler_name, budget, steps):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    manifest = build_manifest()
    shape = manifest["scheduler_shapes"][scheduler_name]
    infimum_lr, params = scheduler_parameters(shape, 1.0, budget)
    extras = {
        key: value
        for key, value in params.items()
        if key
        not in {
            "peak_lr",
            "infimum_lr",
            "max_iter",
            "infimum_ratio",
            "warmup_fraction",
        }
    }
    scheduler = create_scheduler(
        scheduler_name,
        optimizer,
        1.0,
        infimum_lr,
        budget,
        steps_per_epoch=1,
        **extras,
    )
    values = [optimizer.param_groups[0]["lr"]]
    for _ in range(steps):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    return values, params


def test_protocol_manifest_is_self_consistent_and_seed_unique():
    manifest = build_manifest()
    validate_manifest(manifest)
    assert manifest["metrics"]["regression_direction"] == "minimize"
    assert manifest["metrics"]["classification_direction"] == "maximize"
    assert TASK_SCORE_DIRECTIONS == {"integral": "minimize", "cifar10": "maximize"}
    assert final_window(10) == 3
    assert final_window(200) == 20

    for optimizer, spec in GRID_SPECS.items():
        grid = candidate_grid(optimizer)
        assert len(grid) == 30
        assert grid[0]["lr"] == pytest.approx(spec["lower"])
        assert grid[-1]["lr"] == pytest.approx(spec["upper"])
        assert all(left["lr"] < right["lr"] for left, right in itertools.pairwise(grid))

    seeds = []
    for dataset in manifest["seeds"]["datasets"].values():
        seeds.extend(dataset.values())
    for setting in manifest["seeds"]["settings"].values():
        for replicate in setting:
            seeds.append(replicate["hpo_train_seed"])
            seeds.extend(replicate["deployment_seeds"])
    assert len(seeds) == len(set(seeds))


def test_protocol_rejects_candidate_grid_tampering():
    manifest = build_manifest()
    manifest["candidate_grids"]["adamw"]["candidates"][0]["lr"] *= 2
    unhashed = {key: value for key, value in manifest.items() if key != "protocol_hash"}

    manifest["protocol_hash"] = content_hash(unhashed)
    with pytest.raises(ValueError, match="candidate grid hash mismatch"):
        validate_manifest(manifest)


def test_all_core_schedulers_follow_frozen_parameter_contract():
    manifest = build_manifest()
    for scheduler_name in CORE_SCHEDULERS:
        values, params = _lr_curve(scheduler_name, budget=200, steps=200)
        expected = {
            "peak_lr": 1.0,
            "infimum_lr": manifest["scheduler_shapes"][scheduler_name].get(
                "infimum_ratio", 0.0
            ),
            "max_iter": 200,
            **manifest["scheduler_shapes"][scheduler_name],
        }
        assert params == expected
        assert len(values) == 201
        assert all(math.isfinite(value) and value >= 0.0 for value in values)

    d2z, _ = _lr_curve("D2Z", budget=200, steps=200)
    cosine, _ = _lr_curve("C", budget=200, steps=200)
    assert d2z[-1] == pytest.approx(0.0, abs=1e-15)
    assert cosine[-1] == pytest.approx(1e-3, rel=1e-12)


def test_horizon_free_baselines_share_their_pre_cooldown_prefix():
    for scheduler_name in ("POW", "WSQD"):
        short, _ = _lr_curve(scheduler_name, budget=50, steps=20)
        long, _ = _lr_curve(scheduler_name, budget=200, steps=20)
        assert short == pytest.approx(long, rel=0.0, abs=0.0)


def test_production_job_matrix_counts_and_lane_partition(tmp_path):
    protocol = build_manifest()
    pilot_hpo = build_jobs(protocol, tmp_path, "hpo", "S1", "pilot", (10, 200))
    full_hpo = build_jobs(protocol, tmp_path, "hpo", "S1", "full", (10, 50, 100, 200))
    arm_a = build_jobs(protocol, tmp_path, "arm-a", "S1", "pilot", (10, 50, 100, 200))
    assert len(pilot_hpo) == 8 * 30
    assert len(full_hpo) == 4 * 12 * 30 + 4 * 3 * 30
    assert len(arm_a) == 8 * 30 * 3 * 4

    left = lane_jobs(arm_a, 0, 2)
    right = lane_jobs(arm_a, 1, 2)
    assert len(left) == len(right) == len(arm_a) // 2
    assert set(left).isdisjoint(right)
    assert set(left) | set(right) == set(arm_a)


def test_data_file_validation_rejects_tampering(tmp_path):
    generation = tmp_path / "generation.json"
    generation.write_text("{}\n", encoding="utf-8")
    splits = {}
    for name in ("train", "validation", "test"):
        path = tmp_path / f"{name}.parquet"
        path.write_bytes(name.encode())
        splits[name] = {"path": path.name, "sha256": file_sha256(path)}
    manifest = {
        "kind": "operator",
        "generation_sha256": file_sha256(generation),
        "splits": splits,
    }
    validate_data_files(tmp_path, manifest)
    (tmp_path / "train.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_data_files(tmp_path, manifest)


def test_candidate_selection_respects_metric_direction_and_tie_break(tmp_path):
    root = tmp_path / "dry-hpo" / "S1" / "EH"
    root.mkdir(parents=True)
    rows = [
        {"candidate_index": 0, "validation_metric": 0.7},
        {"candidate_index": 1, "validation_metric": 0.2},
        {"candidate_index": 2, "validation_metric": 0.2},
        {"candidate_index": 3, "validation_metric": "nan"},
    ]
    for row in rows:
        (root / f"{row['candidate_index']}.summary.json").write_text(
            json.dumps(row), encoding="utf-8"
        )

    assert selected_candidate(tmp_path, "EH", len(rows), "integral") == 1
    assert selected_candidate(tmp_path, "EH", len(rows), "cifar10") == 0


def test_exact_sign_flip_and_holm_correction():
    assert exact_sign_flip_pvalue([1.0, 2.0, 3.0]) == pytest.approx(1 / 8)
    assert holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04}) == pytest.approx(
        {"a": 0.03, "b": 0.06, "c": 0.06}
    )


def test_nonfinite_metrics_have_strict_json_tokens():
    assert metric_token(None) is None
    assert metric_token(float("nan")) == "nan"
    assert metric_token(float("inf")) == "inf"
    assert metric_token(float("-inf")) == "-inf"
