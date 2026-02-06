#!/bin/bash
# Install top-level dependencies extracted from requirements.txt
# Transitive dependencies (pulled in via other packages) are omitted
# as they will be resolved automatically by uv.

set -e

uv pip install -U \
    einops \
    joblib \
    matplotlib \
    numpy \
    optuna \
    packaging \
    pandas \
    polars \
    pyarrow \
    scienceplots \
    scipy \
    survey \
    torch \
    torchvision \
    tqdm \
    wandb
