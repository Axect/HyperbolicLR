"""Run one frozen v13 HPO candidate, deployment, Arm A, or dry-pipeline job."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import polars as pl
import torch
import torch.nn.functional as F
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from confirmatory_data import load_data_manifest, validate_data_files
from confirmatory_protocol import (
    CORE_SCHEDULERS,
    HORIZONS,
    TASK_SCORE_DIRECTIONS,
    TUNE_BUDGET,
    candidate_grid_hash,
    canonical_json,
    content_hash,
    file_sha256,
    final_window,
    load_manifest,
    make_run_id,
    make_selection_id,
    metric_token,
    replicate_count,
)
from model import DeepONet, DeepONetLN, ResNetCIFAR, SimpleViT, TFONet
from scheduler_config import create_scheduler, is_step_per_batch
from util import (
    OperatorTrainer,
    Trainer,
    load_cifar10_confirmatory,
    load_integral_confirmatory,
    set_seed,
)

STAGES = ("hpo", "deploy", "arm-a", "dry-hpo", "dry-deploy")


def git_provenance(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    import random

    import numpy as np

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    dataset, batch_size: int, shuffle: bool, seed: int, device: str
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    workers = 0 if isinstance(dataset, TensorDataset) else 2
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
        generator=generator,
        worker_init_fn=seed_worker if workers else None,
        persistent_workers=bool(workers),
    )


def dry_data(
    seed: int, include_test: bool, device: str
) -> tuple[Any, DataLoader, DataLoader, DataLoader | None]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(48, 4, generator=generator)
    target = (0.7 * x[:, :1] - 0.2 * x[:, 1:2] + 0.1 * x[:, 2:3]).tanh()
    train = TensorDataset(x[:32], target[:32])
    validation = TensorDataset(x[32:40], target[32:40])
    test = TensorDataset(x[40:], target[40:]) if include_test else None
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    return (
        model,
        make_loader(train, 32, True, seed, device),
        make_loader(validation, 8, False, seed, device),
        make_loader(test, 8, False, seed, device) if test is not None else None,
    )


def build_real_data_and_model(
    setting: dict[str, Any],
    data_root: Path,
    split_seed: int,
    train_seed: int,
    include_test: bool,
    device: str,
):
    batch_size = int(setting["batch_size"])
    if setting["task"] == "integral":
        splits = load_integral_confirmatory(data_root, include_test=include_test)
        datasets = {name: TensorDataset(*tensors) for name, tensors in splits.items()}
        model_class = {
            "DeepONet": DeepONet,
            "DeepONetLN": DeepONetLN,
            "TFONet": TFONet,
        }[setting["model"]]
        model = model_class(model_class.default_hparams, device=device)
        trainer_class = OperatorTrainer
    elif setting["task"] == "cifar10":
        train, validation, test, _ = load_cifar10_confirmatory(
            split_seed=split_seed,
            root=str(data_root),
            download=False,
            include_test=include_test,
        )
        datasets = {"train": train, "validation": validation}
        if test is not None:
            datasets["test"] = test
        if setting["model"] == "ResNetCIFAR18":
            model = ResNetCIFAR({"depth": 18}, num_classes=10, device=device)
        elif setting["model"] == "SimpleViT":
            model = SimpleViT(SimpleViT.default_hparams, num_classes=10, device=device)
        else:
            raise ValueError(f"unsupported CIFAR model: {setting['model']}")
        trainer_class = Trainer
    else:
        raise ValueError(f"unsupported task: {setting['task']}")

    train_loader = make_loader(datasets["train"], batch_size, True, train_seed, device)
    validation_loader = make_loader(
        datasets["validation"], batch_size, False, train_seed, device
    )
    test_loader = None
    if include_test:
        test_loader = make_loader(
            datasets["test"], batch_size, False, train_seed, device
        )
    return model, train_loader, validation_loader, test_loader, trainer_class


def build_optimizer(setting: dict[str, Any], model: nn.Module, lr: float):
    params = dict(setting["optimizer_params"])
    if setting["optimizer"] == "AdamW":
        params["betas"] = tuple(params["betas"])
        return optim.AdamW(model.parameters(), lr=lr, **params)
    if setting["optimizer"] == "SGD":
        return optim.SGD(model.parameters(), lr=lr, **params)
    raise ValueError(f"unsupported optimizer: {setting['optimizer']}")


def scheduler_parameters(
    shape: dict[str, Any], lr: float, budget: int
) -> tuple[float, dict[str, Any]]:
    infimum_ratio = shape.get("infimum_ratio")
    infimum_lr = lr * infimum_ratio if infimum_ratio is not None else 0.0
    return infimum_lr, {
        "peak_lr": lr,
        "infimum_lr": infimum_lr,
        "max_iter": budget,
        **shape,
    }


def evaluate_test(trainer, loader, classification: bool) -> tuple[float, float]:
    result = trainer.evaluate(loader)
    if classification:
        loss, accuracy = result
        return float(loss), float(accuracy)
    loss = float(result)
    return loss, loss


def train_job(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader | None,
    setting: dict[str, Any],
    scheduler_name: str,
    lr: float,
    budget: int,
    shape: dict[str, Any],
    device: str,
) -> tuple[list[dict[str, Any]], str]:
    optimizer = build_optimizer(setting, model, lr)
    infimum_lr, actual_scheduler_params = scheduler_parameters(shape, lr, budget)
    extras = {
        key: value
        for key, value in actual_scheduler_params.items()
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
        lr,
        infimum_lr,
        budget,
        steps_per_epoch=len(train_loader),
        **extras,
    )
    classification = setting["task"] == "cifar10"
    if classification:
        trainer = Trainer(
            model,
            optimizer,
            scheduler,
            criterion=F.cross_entropy,
            acc=True,
            device=device,
            step_per_batch=is_step_per_batch(scheduler_name),
        )
    elif setting["task"] == "integral":
        trainer = OperatorTrainer(
            model,
            optimizer,
            scheduler,
            device=device,
            step_per_batch=is_step_per_batch(scheduler_name),
        )
    else:
        trainer = Trainer(
            model,
            optimizer,
            scheduler,
            criterion=F.mse_loss,
            acc=False,
            device=device,
            step_per_batch=is_step_per_batch(scheduler_name),
        )

    first_test_epoch = budget - final_window(budget) + 1

    def callback(epoch: int, _val_loss: float) -> None:
        if test_loader is None or epoch < first_test_epoch:
            return
        test_loss, test_score = evaluate_test(trainer, test_loader, classification)
        trainer.history[-1]["test_loss"] = test_loss
        trainer.history[-1]["test_score"] = test_score

    trainer.train(
        train_loader, validation_loader, epochs=budget, epoch_callback=callback
    )
    for row in trainer.history:
        row["val_score"] = float(row["acc"] if classification else row["val_loss"])
        row.setdefault("test_loss", None)
        row.setdefault("test_score", None)
        row.pop("acc", None)
    return trainer.history, canonical_json(actual_scheduler_params)


def final_metric(
    history: list[dict[str, Any]], split: str, budget: int, classification: bool
) -> float:
    key = f"{split}_score"
    values = [
        row[key] for row in history[-final_window(budget) :] if row[key] is not None
    ]
    if len(values) != final_window(budget):
        raise RuntimeError(
            f"expected {final_window(budget)} {split} scores, found {len(values)}"
        )
    values.sort()
    middle = len(values) // 2
    median = (
        values[middle]
        if len(values) % 2
        else 0.5 * (values[middle - 1] + values[middle])
    )
    if classification:
        return float(median)
    return float(median)


def add_metadata(
    history: list[dict[str, Any]], metadata: dict[str, Any]
) -> pl.DataFrame:
    frame = pl.DataFrame(history, infer_schema_length=None)
    for key, value in metadata.items():
        frame = frame.with_columns(pl.lit(value).alias(key))
    return frame


def resolve_seed(
    protocol: dict[str, Any],
    setting_id: str,
    stage: str,
    replicate: int,
    seed_slot: int,
):
    record = protocol["seeds"]["settings"][setting_id][replicate]
    if stage in {"hpo", "dry-hpo"}:
        return int(record["hpo_train_seed"]), None
    deployment_seed = int(record["deployment_seeds"][seed_slot])
    return int(record["hpo_train_seed"]), deployment_seed


def run(args: argparse.Namespace) -> Path:
    protocol = load_manifest(args.protocol)
    if args.scheduler not in CORE_SCHEDULERS:
        raise ValueError(f"scheduler is outside frozen core: {args.scheduler}")
    if args.setting not in protocol["settings"]:
        raise ValueError(f"unknown setting: {args.setting}")
    setting = protocol["settings"][args.setting]
    grid = protocol["candidate_grids"][setting["optimizer_key"]]
    if grid["sha256"] != candidate_grid_hash(setting["optimizer_key"]):
        raise ValueError("candidate grid manifest does not match source")
    if not 0 <= args.candidate_index < len(grid["candidates"]):
        raise ValueError("candidate index outside frozen grid")
    if args.stage in {"hpo", "dry-hpo"} and args.budget != TUNE_BUDGET:
        raise ValueError(f"HPO budget must be {TUNE_BUDGET}")
    if args.stage in {"deploy", "arm-a", "dry-deploy"} and not 0 <= args.seed_slot < 3:
        raise ValueError("deployment and Arm A jobs require seed-slot 0, 1, or 2")
    if not 0 <= args.hpo_replicate < replicate_count(args.scheduler):
        raise ValueError(
            f"replicate {args.hpo_replicate} is outside the frozen count for {args.scheduler}"
        )
    if args.stage == "arm-a" and args.hpo_replicate != 0:
        raise ValueError("Arm A uses replicate 0 deployment seeds")
    if args.stage in {"deploy", "arm-a", "dry-deploy"} and args.budget not in HORIZONS:
        raise ValueError(f"deployment budget must be one of {HORIZONS}")

    data_manifest = load_data_manifest(args.data_manifest, protocol["protocol_hash"])
    expected_group = (
        "dry-v13" if args.stage.startswith("dry-") else setting["dataset_group"]
    )
    if data_manifest["dataset_group"] != expected_group:
        raise ValueError(
            f"data group mismatch: {data_manifest['dataset_group']} != {expected_group}"
        )
    if not args.stage.startswith("dry-"):
        validate_data_files(args.data_root, data_manifest)

    hpo_seed, deployment_seed = resolve_seed(
        protocol, args.setting, args.stage, args.hpo_replicate, args.seed_slot
    )
    train_seed = deployment_seed if deployment_seed is not None else hpo_seed
    include_test = args.stage in {"deploy", "arm-a", "dry-deploy"}
    lr = float(grid["candidates"][args.candidate_index]["lr"])
    selection_id = make_selection_id(
        protocol["protocol_hash"], args.setting, args.scheduler, args.hpo_replicate
    )
    run_id = make_run_id(
        protocol["protocol_hash"],
        args.stage,
        args.setting,
        args.scheduler,
        args.hpo_replicate,
        args.candidate_index,
        args.budget,
        train_seed,
    )
    output = (
        args.output_root
        / args.stage
        / args.setting
        / args.scheduler
        / f"{run_id}.parquet"
    )
    summary_path = output.with_suffix(".summary.json")
    if output.exists():
        existing = pl.read_parquet(output, columns=["run_id", "protocol_hash"]).row(
            0, named=True
        )
        if existing != {"run_id": run_id, "protocol_hash": protocol["protocol_hash"]}:
            raise RuntimeError(f"existing artifact identity mismatch: {output}")
        if not summary_path.is_file():
            raise RuntimeError(f"curve exists without summary: {output}")
        print(f"skip verified: {output}")
        return output

    repo_root = Path(__file__).resolve().parent.parent
    commit, dirty = git_provenance(repo_root)
    if dirty and not (args.allow_dirty and args.stage.startswith("dry-")):
        raise RuntimeError("production run requires a clean Git working tree")
    lock_path = Path(__file__).resolve().parent / "requirements.txt"
    environment_lock_hash = file_sha256(lock_path)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    set_seed(train_seed)
    torch.use_deterministic_algorithms(True)
    if args.stage.startswith("dry-"):
        torch.set_num_threads(1)
        model, train_loader, validation_loader, test_loader = dry_data(
            train_seed, include_test, device
        )
        dry_setting = dict(setting)
        dry_setting["task"] = "dry"
        dry_setting["optimizer"] = "AdamW"
        dry_setting["optimizer_params"] = {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        }
        dry_setting["batch_size"] = 32
        run_setting = dry_setting
    else:
        dataset_seeds = protocol["seeds"]["datasets"][setting["dataset_group"]]
        model, train_loader, validation_loader, test_loader, _ = (
            build_real_data_and_model(
                setting,
                args.data_root,
                int(dataset_seeds["split"]),
                train_seed,
                include_test,
                device,
            )
        )
        run_setting = setting

    wandb.init(mode="disabled")
    started = time.perf_counter()
    history, scheduler_params = train_job(
        model,
        train_loader,
        validation_loader,
        test_loader,
        run_setting,
        args.scheduler,
        lr,
        args.budget,
        protocol["scheduler_shapes"][args.scheduler],
        device,
    )
    elapsed = time.perf_counter() - started
    classification = run_setting["task"] == "cifar10"
    validation_metric = final_metric(history, "val", args.budget, classification)
    test_metric = (
        final_metric(history, "test", args.budget, classification)
        if include_test
        else None
    )
    finite = math.isfinite(validation_metric) and (
        test_metric is None or math.isfinite(test_metric)
    )
    method_outcome = "finite" if finite else "nonfinite"

    metadata = {
        "run_id": run_id,
        "selection_id": selection_id,
        "protocol_hash": protocol["protocol_hash"],
        "stage": args.stage,
        "setting": args.setting,
        "task": setting["task"],
        "model": setting["model"],
        "scheduler": args.scheduler,
        "hpo_replicate": args.hpo_replicate,
        "candidate_index": args.candidate_index,
        "hpo_train_seed": hpo_seed,
        "deployment_seed": deployment_seed,
        "split_hash": data_manifest["split_hash"],
        "candidate_grid_hash": grid["sha256"],
        "optimizer": setting["optimizer"],
        "optimizer_params": canonical_json(setting["optimizer_params"]),
        "scheduler_params": scheduler_params,
        "tune_budget": TUNE_BUDGET,
        "budget": args.budget,
        "lr0": lr,
        "code_commit": commit,
        "dirty": dirty,
        "environment_lock_hash": environment_lock_hash,
        "device": device,
        "run_status": "complete",
        "method_outcome": method_outcome,
        "test_accessed": include_test,
        "metric_direction": TASK_SCORE_DIRECTIONS[setting["task"]],
    }
    frame = add_metadata(history, metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.partial")
    frame.write_parquet(temporary)
    temporary.replace(output)
    summary = {
        **metadata,
        "epochs": len(history),
        "validation_metric": metric_token(validation_metric),
        "test_metric": metric_token(test_metric),
        "elapsed_seconds": elapsed,
        "curve_sha256": file_sha256(output),
        "metric_hash": content_hash(
            {
                "val_score": [metric_token(row["val_score"]) for row in history],
                "test_score": [metric_token(row["test_score"]) for row in history],
            }
        ),
    }
    atomic_write_json(summary_path, summary)
    wandb.finish()
    print(
        f"wrote {output} validation={validation_metric:.6g} "
        f"test={test_metric if test_metric is not None else 'not-accessed'} outcome={method_outcome}"
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument(
        "--setting",
        choices=tuple(sorted(("S1", "S2", "S3", "S4", "S5"))),
        required=True,
    )
    parser.add_argument("--scheduler", choices=CORE_SCHEDULERS, required=True)
    parser.add_argument("--hpo-replicate", type=int, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed-slot", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
