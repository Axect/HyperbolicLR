"""Build and validate data manifests for the v13 confirmatory campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from confirmatory_protocol import content_hash, file_sha256, load_manifest
from util import load_cifar10_confirmatory


def _write_manifest(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in manifest.items() if key != "split_hash"}
    manifest["split_hash"] = content_hash(unsigned)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_data_manifest(path: str | Path, protocol_hash: str) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed = manifest.get("split_hash")
    unsigned = {key: value for key, value in manifest.items() if key != "split_hash"}
    actual = content_hash(unsigned)
    if claimed != actual:
        raise ValueError(
            f"data manifest hash mismatch: claimed={claimed}, actual={actual}"
        )
    if manifest.get("protocol_hash") != protocol_hash:
        raise ValueError("data manifest belongs to a different protocol")
    return manifest


def validate_data_files(data_root: Path, manifest: dict[str, Any]) -> None:
    kind = manifest["kind"]
    if kind == "operator":
        generation_path = data_root / "generation.json"
        if file_sha256(generation_path) != manifest["generation_sha256"]:
            raise ValueError("operator generation metadata checksum mismatch")
        for split in manifest["splits"].values():
            path = data_root / split["path"]
            if file_sha256(path) != split["sha256"]:
                raise ValueError(f"operator split checksum mismatch: {path}")
        return
    if kind == "classification":
        source_root = data_root / "cifar-10-batches-py"
        for name, expected in manifest["source_files"].items():
            path = source_root / name
            if file_sha256(path) != expected:
                raise ValueError(f"CIFAR-10 source checksum mismatch: {path}")
        return
    if kind != "synthetic-regression":
        raise ValueError(f"unknown data-manifest kind: {kind}")


def build_operator_manifest(
    data_root: Path, protocol: dict[str, Any], output: Path
) -> dict[str, Any]:
    generation_path = data_root / "generation.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    expected_seeds = protocol["seeds"]["datasets"]["operator-grf-v13"]
    expected = {
        "train_count": 8_000,
        "validation_count": 1_000,
        "test_count": 1_000,
        "train_seed": expected_seeds["train"],
        "validation_seed": expected_seeds["validation"],
        "test_seed": expected_seeds["test"],
    }
    for key, value in expected.items():
        if generation.get(key) != value:
            raise ValueError(
                f"operator generation mismatch for {key}: {generation.get(key)} != {value}"
            )

    splits: dict[str, Any] = {}
    for name, count in (("train", 8_000), ("validation", 1_000), ("test", 1_000)):
        path = data_root / f"{name}.parquet"
        frame = pl.read_parquet(path)
        if frame.columns != ["u", "y", "Gu"]:
            raise ValueError(f"unexpected columns in {path}: {frame.columns}")
        if frame.height != count * 100:
            raise ValueError(
                f"unexpected row count in {path}: {frame.height} != {count * 100}"
            )
        splits[name] = {
            "count": count,
            "sample_width": 100,
            "path": path.name,
            "sha256": file_sha256(path),
        }

    manifest = {
        "protocol_hash": protocol["protocol_hash"],
        "dataset_group": "operator-grf-v13",
        "kind": "operator",
        "generation": generation,
        "generation_sha256": file_sha256(generation_path),
        "splits": splits,
        "transform": {
            "grf_scaling": "training extrema mean plus or minus one training standard deviation",
            "fit_split": "train",
            "model_input_standardization": None,
        },
    }
    return _write_manifest(output, manifest)


def _indices_hash(indices: list[int]) -> str:
    return content_hash(indices)


def build_cifar10_manifest(
    data_root: Path,
    protocol: dict[str, Any],
    output: Path,
    download: bool,
) -> dict[str, Any]:
    split_seed = protocol["seeds"]["datasets"]["cifar10-v13"]["split"]
    train, validation, test, split = load_cifar10_confirmatory(
        split_seed=split_seed, root=str(data_root), download=download
    )
    source_root = data_root / "cifar-10-batches-py"
    source_names = [f"data_batch_{index}" for index in range(1, 6)] + [
        "test_batch",
        "batches.meta",
    ]
    source_files = {}
    for name in source_names:
        path = source_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        source_files[name] = file_sha256(path)

    manifest = {
        "protocol_hash": protocol["protocol_hash"],
        "dataset_group": "cifar10-v13",
        "kind": "classification",
        "source": "torchvision CIFAR10",
        "source_files": source_files,
        "split_seed": split_seed,
        "splits": {
            "train": {
                "count": len(train),
                "indices_sha256": _indices_hash(split["train_indices"]),
            },
            "validation": {
                "count": len(validation),
                "indices_sha256": _indices_hash(split["validation_indices"]),
            },
            "test": {
                "count": len(test),
                "indices_sha256": _indices_hash(split["test_indices"]),
            },
        },
        "transforms": {
            "train": [
                "RandomCrop(32,padding=4)",
                "RandomHorizontalFlip()",
                "ToTensor()",
                "Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))",
            ],
            "validation_and_test": [
                "ToTensor()",
                "Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))",
            ],
        },
    }
    return _write_manifest(output, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("operator", "cifar10", "validate"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    protocol = load_manifest(args.protocol)
    if args.kind == "validate":
        if args.manifest is None or args.data_root is None:
            parser.error("validate requires --manifest and --data-root")
        manifest = load_data_manifest(args.manifest, protocol["protocol_hash"])
        validate_data_files(args.data_root, manifest)
        print(
            f"valid {args.manifest} split_hash={manifest['split_hash']} files=verified"
        )
        return
    if args.data_root is None or args.output is None:
        parser.error(f"{args.kind} requires --data-root and --output")
    if args.kind == "operator":
        manifest = build_operator_manifest(args.data_root, protocol, args.output)
    else:
        manifest = build_cifar10_manifest(
            args.data_root, protocol, args.output, args.download
        )
    print(f"wrote {args.output} split_hash={manifest['split_hash']}")


if __name__ == "__main__":
    main()
