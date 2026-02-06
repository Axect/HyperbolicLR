# HyperbolicLR: Epoch-insensitive Learning Rate Scheduler - Experiments

This directory contains the experimental code, data, and analysis tools for the HyperbolicLR paper. It includes implementations of various deep learning models, data processing utilities, and scripts for running experiments and analyzing results.

## Directory Structure

- `analyze/`: Rust-based analysis tools for experimental results
- `data/`: Dataset files
- `figs/`: Generated figures and plots
- `learning_curve_example/`: Rust-based learning curve analysis
- `learning_curves/`: Rust-based learning curve plotting per scheduler
- `metric/`: Rust-based metric calculation tools
- `osc/`: Rust-based oscillation data generation
- `scripts/`: Various Python scripts for plotting and analysis

## Main Python Files

### `scheduler_config.py`

Centralized scheduler configuration module. All scheduler-related logic (registry, parameter defaults, Optuna search spaces, factory creation) is defined here. The three `main_*.py` files import from this module, eliminating code duplication.

Supported schedulers:

| Code | Scheduler | Type |
|------|-----------|------|
| `N` | No Scheduler | Baseline |
| `P` | PolynomialLR | Traditional |
| `C` | CosineAnnealingLR | Traditional |
| `E` | ExponentialLR | Traditional |
| `L` | LinearLR | Traditional |
| `S` | StepLR | Traditional |
| `OC` | OneCycleLR | Cyclic (per-batch) |
| `CY` | CyclicLR | Cyclic (per-batch) |
| `H` | HyperbolicLR | Hyperbolic |
| `EH` | ExpHyperbolicLR | Hyperbolic |
| `WH` | Warmup + HyperbolicLR | Hyperbolic + Warmup |
| `WEH` | Warmup + ExpHyperbolicLR | Hyperbolic + Warmup |
| `WC` | Warmup + CosineAnnealingLR | Traditional + Warmup |

LR-free optimizers (Compare mode):

| Name | Optimizer | Recommended LR |
|------|-----------|----------------|
| `Prodigy` | Prodigy | 1.0 |
| `DAdapt` | DAdaptAdam | 1.0 |

### `main_cifar10.py`

CIFAR-10/100 image classification experiments.

- Models: `SimpleCNN`, `ResNetCIFAR`, `SimpleViT`
- Run modes: Run, Compare, Optimize, Optimize-Compare
- Integration with Weights & Biases for experiment tracking

### `main_osc.py`

Oscillation prediction experiments with LSTM Sequence-to-Sequence model.

- Data types: simple, damped, total
- Run modes: Run, Compare, Optimize, Optimize-Compare

### `main_integral.py`

Integral operator learning experiments.

- Models: `DeepONet`, `TFONet`
- Run modes: Run, Compare, Optimize, Optimize-Compare

### `run_all.py`

Non-interactive All-in-One experiment runner. Automates the full Optimize → Optimize-Compare pipeline for all schedulers and LR-free optimizers via CLI arguments only.

- Phase 1: Optuna TPE optimization for every scheduler
- Phase 2: Epoch sensitivity comparison with best params
- Saves results to `results_{project}.json`

### `model.py`

Neural network architectures:

- `SimpleCNN`: Basic convolutional neural network
- `ResNetCIFAR`: ResNet-18/34/50 adapted for CIFAR 32x32 images (3x3 conv stem, no maxpool)
- `SimpleViT`: Lightweight Vision Transformer (patch_size=4, embed_dim=192, depth=6)
- `LSTM_Seq2Seq`: LSTM-based sequence-to-sequence model
- `DeepONet`: Deep Operator Network
- `TFONet`: Transformer-based Operator Network

### `util.py`

Data loading and training utilities:

- `load_cifar10()`, `load_cifar100()`: CIFAR data loading with optional subsetting (`subset_ratio >= 1.0` skips subsetting)
- `Trainer`: General-purpose trainer with `step_per_batch` support for OneCycleLR/CyclicLR
- `OperatorTrainer`: Trainer for operator learning models with `step_per_batch` support

### `result_plot.py`

Result visualization with grouped subplot design (Traditional / Cyclic+Warmup / Hyperbolic / LR-free).

### `metric_plot.py`

Standalone LR curve visualization for all scheduler types.

## Requirements

See `requirements.txt` for the list of Python dependencies.

For Rust components, ensure you have Rust installed. Dependencies are managed via Cargo.

## Setup

```sh
cd paper

# UV (Recommended)
uv venv
uv pip sync requirements.txt

# Install LR-free optimizer dependencies
uv pip install prodigyopt dadaptation

# Activate
source .venv/bin/activate
```

## Data Generation

Before running experiments, generate the required datasets:

```sh
# Oscillation data
cd osc && cargo run --release && cd ..

# Integral data
cd integral && cargo run --release && cd ..

# CIFAR-10/100 is downloaded automatically on first run
```

## Running Experiments

All experiment scripts use an interactive CLI (via `survey`). Run them with `uv`:

```sh
uv run python main_cifar10.py
uv run python main_osc.py
uv run python main_integral.py
```

### Run Modes (Interactive Scripts)

Each `main_*.py` script supports 4 interactive run modes:

| Mode | Description |
|------|-------------|
| **Run** | Single run with manually specified hyperparameters |
| **Compare** | Run all scheduler x optimizer combinations with default parameters |
| **Optimize** | Optuna hyperparameter optimization (25 trials) for a single scheduler |
| **Optimize-Compare** | Load Optuna best trial and run across epoch budgets [50, 100, 150, 200] |

### All-in-One Workflow (Recommended)

The `run_all.py` script automates the entire Optimize → Optimize-Compare pipeline non-interactively:

```sh
# Full experiment: all schedulers + LR-free optimizers
uv run python run_all.py --task cifar10 --model ResNetCIFAR

# Quick dry run (sanity check)
uv run python run_all.py --task cifar10 --model SimpleCNN \
  --epochs 2 --n-trials 2 --n-seeds 1 --wandb-mode disabled

# Other tasks
uv run python run_all.py --task osc
uv run python run_all.py --task integral --model DeepONet

# Customize
uv run python run_all.py --task cifar10 --model ResNetCIFAR \
  --n-trials 10 --epochs 100 --schedulers EH,WEH,C --skip-lr-free
```

Key options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `cifar10` | Task: `cifar10`, `osc`, `integral` |
| `--model` | (task default) | Model name |
| `--optimizer` | `AdamW` | Base optimizer: `Adam`, `AdamW` |
| `--epochs` | `50` | Optimization epoch budget (low for epoch-insensitivity test) |
| `--n-trials` | `25` | Optuna trials per scheduler |
| `--n-seeds` | `1` | Seeds during optimization (1=fast, 5=robust) |
| `--compare-seeds` | `5` | Seeds during comparison phase |
| `--epoch-budgets` | `50,100,150,200` | Epoch budgets for comparison (ascending) |
| `--schedulers` | (all) | Comma-separated subset of schedulers |
| `--skip-lr-free` | `False` | Skip LR-free optimizers |
| `--skip-compare` | `False` | Skip Phase 2 (only optimize) |
| `--wandb-mode` | `online` | W&B mode: `online`, `offline`, `disabled` |

### Interactive Workflow

For individual experiments, use the interactive scripts directly:

#### Step 1: Pilot Run (Sanity Check)

```sh
uv run python main_cifar10.py
# Select: Run mode → CIFAR10 → subset_ratio=0.1 → SimpleCNN → batch_size=128 → epochs=10
```

#### Step 2: Optimize Hyperparameters

```sh
uv run python main_cifar10.py --project HyperbolicLR-CIFAR10-ResNet
# Select: Optimize mode → CIFAR10 → subset_ratio=1.0 → ResNetCIFAR → batch_size=128 → epochs=200
# Then select optimizer (AdamW) and scheduler (e.g., EH)
```

#### Step 3: Epoch Sensitivity Comparison

```sh
uv run python main_cifar10.py --project HyperbolicLR-CIFAR10-ResNet
# Select: Optimize-Compare mode → select study → runs epochs=[200, 150, 100, 50]
```

#### Step 4: Direct Comparison

```sh
uv run python main_cifar10.py --project HyperbolicLR-CIFAR10-ResNet
# Select: Compare mode → runs 2 optimizers × 13 schedulers + 2 LR-free = 28 runs
```

### Full Experiment Matrix

To reproduce all paper results, run for each combination:

**CIFAR-10 (Image Classification)**
| Model | subset_ratio | batch_size | epochs | lr | infimum_lr |
|-------|-------------|------------|--------|-----|-----------|
| SimpleCNN | 0.1 | 128 | 200 | 0.01 | 1e-6 |
| ResNetCIFAR | 1.0 | 128 | 200 | 0.01 | 1e-6 |
| SimpleViT | 1.0 | 128 | 200 | 0.001 | 1e-6 |

**OSC (Time Series Prediction)**
| Model | dtype | hist | pred | batch_size | epochs | lr | infimum_lr |
|-------|-------|------|------|------------|--------|-----|-----------|
| LSTM_Seq2Seq | simple | 50 | 50 | 64 | 200 | 0.01 | 1e-6 |

**Integral (Operator Learning)**
| Model | batch_size | epochs | lr | infimum_lr |
|-------|------------|--------|-----|-----------|
| DeepONet | 128 | 200 | 0.005 | 1e-6 |
| TFONet | 128 | 200 | 0.001 | 1e-6 |

### Notes on Specific Schedulers

- **OneCycleLR (`OC`) / CyclicLR (`CY`)**: These schedulers step per-batch, not per-epoch. The `step_per_batch` flag is automatically set via `is_step_per_batch()` in `scheduler_config.py`.
- **Warmup variants (`WH`, `WEH`, `WC`)**: Warmup duration defaults to `epochs // 10`. Particularly important for ResNetCIFAR (BatchNorm) and SimpleViT (Transformer).
- **LR-free optimizers**: Prodigy and DAdaptAdam use `lr=1.0` and self-tune internally. They run without a scheduler (`N`).

## Analysis

After running experiments, results are logged to W&B. To generate analysis outputs:

### 1. Export W&B Data

Export learning curves from W&B to CSV format (for Rust analysis tools):

```
data/Result_{Dataset}_{Model}-{Scheduler}.csv
```

Each CSV should contain columns like `{Scheduler}{Epochs}_val_loss` (e.g., `EH200_val_loss`).

### 2. Rust Analysis Tools

```sh
# Learning curve difference analysis (SLCD metric)
cd analyze && cargo run --release && cd ..

# Accuracy-specific analysis
cd analyze && cargo run --release --bin acc && cd ..

# Learning curve plots per scheduler
cd learning_curves && cargo run --release && cd ..

# LR scheduler metric evaluation
cd metric && cargo run --release && cd ..
```

### 3. Python Visualization

```sh
# LR schedule curves (all schedulers)
uv run python metric_plot.py

# Result bar charts (grouped by scheduler family)
uv run python result_plot.py

# Decay rate analysis (d(eta)/dt comparison)
uv run python scripts/03_decay_rate_analysis.py

# Cumulative LR budget analysis
uv run python scripts/04_cumulative_lr_budget.py
```

## Scheduler Overview

### Traditional Schedulers (Baselines)
- **N** (No Scheduler): Constant LR (lr/10 for fair comparison)
- **L** (LinearLR): Linear decay from `lr` to `infimum_lr`
- **S** (StepLR): Step decay with `gamma=0.1` every `epochs//3` steps
- **P** (PolynomialLR): Polynomial decay with `power=0.5`
- **C** (CosineAnnealingLR): Cosine annealing to `eta_min`
- **E** (ExponentialLR): Exponential decay with `gamma=0.96`

### Cyclic / Warm-up Schedulers
- **OC** (OneCycleLR): Super-convergence schedule (per-batch stepping)
- **CY** (CyclicLR): Triangular cyclic schedule (per-batch stepping)
- **WC** (Warmup + Cosine): `SequentialLR(LinearLR → CosineAnnealingLR)`

### Hyperbolic Schedulers (Ours)
- **H** (HyperbolicLR): Hyperbolic decay with epoch-insensitive property
- **EH** (ExpHyperbolicLR): Exponential variant of HyperbolicLR
- **WH** (Warmup + HyperbolicLR): Linear warmup → hyperbolic decay
- **WEH** (Warmup + ExpHyperbolicLR): Linear warmup → exp-hyperbolic decay

### LR-Free Optimizers
- **Prodigy**: Prodigy optimizer (self-tuning learning rate)
- **DAdapt**: D-Adaptation Adam (learning-rate-free Adam)

## Additional Scripts

### `scripts/01_poly_cos_lr.py`

Generates plots for Polynomial and Cosine Annealing LR schedulers at different epoch settings.

### `scripts/02_prop_1.py`

Generates a plot demonstrating the properties of the hyperbolic function used in HyperbolicLR.

### `scripts/plot_hyperbolic_lr.py`

Creates plots for HyperbolicLR and ExpHyperbolicLR LR curves and initial gradients.

### `scripts/03_decay_rate_analysis.py`

Compares instantaneous decay rate `d(eta)/dt` across all schedulers. Visualizes why hyperbolic decay behaves differently.

### `scripts/04_cumulative_lr_budget.py`

Compares cumulative LR budget `integral(eta, 0, N)` across schedulers for different epoch counts. Demonstrates that HyperbolicLR has similar cumulative budgets across different N.

## Additional Notes

- The `bring.sh` script copies the main `hyperbolic_lr.py` implementation into the experiment directory.
- Rust components provide efficient data generation, analysis, and metric calculation.
- All experiments are tracked via W&B. Set `WANDB_MODE=offline` for offline runs.
- To add a new scheduler, add it only to `scheduler_config.py` — all three `main_*.py` files will pick it up automatically.

For more detailed information on the HyperbolicLR implementation, refer to the main [README](../README.md) in the root directory.
