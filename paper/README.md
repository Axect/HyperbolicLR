



# HyperbolicLR: Epoch-insensitive Learning Rate Scheduler - Experiments

This directory contains the experimental code, data, and analysis tools for the HyperbolicLR paper. It includes implementations of various deep learning models, data processing utilities, and scripts for running experiments and analyzing results.

## Directory Structure

- `analyze/`: Rust-based analysis tools for experimental results
- `data/`: Dataset files
- `figs/`: Generated figures and plots
- `learning_curve_example/`: Rust-based learning curve analysis
- `metric/`: Rust-based metric calculation tools
- `osc/`: Rust-based oscillation data generation
- `scripts/`: Various Python scripts for plotting and analysis

## Main Python Files

### `main_cifar10.py`

This script runs experiments on the CIFAR-10 dataset. It includes:
- Model selection (SimpleCNN or your own model)
- Hyperparameter optimization
- Various run modes (Run, Search, Compare, Optimize, Optimize-Compare)
- Integration with Weights & Biases for experiment tracking

### `main_osc.py`

This script runs experiments on oscillation prediction tasks. Features include:
- LSTM Sequence-to-Sequence model
- Various oscillation data types (simple, damped, total)
- Multiple input/output modes
- Hyperparameter optimization and comparison of different schedulers

### `main_integral.py`

This script runs experiments on integral operator learning tasks. Features include:
- Model selection (DeepONet, TFONet)
- Hyperparameter optimization
- Various run modes (Run, Search, Compare, Optimize, Optimize-Compare)
- Integration with Weights & Biases for experiment tracking

### `model.py`

Defines the neural network architectures used in the experiments:
- `SimpleCNN`: A basic convolutional neural network
- `LSTM_Seq2Seq`: LSTM-based sequence-to-sequence model for time series prediction
- `DeepONet`: Deep Operator Network for learning integral operators
- `TFONet`: Transformer-based Operator Network for learning integral operators

### `util.py`

Utility functions for data loading, preprocessing, and training:
- CIFAR-10 and CIFAR-100 data loading functions
- Oscillation data loading and preprocessing
- Integral data loading function
- `Trainer` class for model training and evaluation
- `OperatorTrainer` class for training operator learning models

### `result_plot.py`

Script for generating plots from experimental results.

## Rust-based Components

### `analyze/`

Contains Rust code for analyzing experimental results:
- `src/main.rs`: Analyzes learning curves for different datasets, models, and schedulers
- `src/bin/acc.rs`: Specifically analyzes accuracy for CIFAR-10 experiments

Features:
- Interactive selection of dataset, model, and scheduler
- Calculation of learning curve differences using Savitzky-Golay filtering
- Generation of comparative statistics

### `metric/`

Implements various learning rate schedulers and evaluates their properties:
- `src/main.rs`: Defines and evaluates PolynomialLR, CosineAnnealingLR, HyperbolicLR, and ExpHyperbolicLR
- Generates plots for each scheduler type
- Calculates and compares metrics across different epoch settings

### `osc/`

Generates oscillation data for experiments:
- `src/main.rs`: Implements a damped simple harmonic oscillator using the Newmark-beta method
- Generates data for different damping ratios
- Produces a plot of the oscillation data
- Saves the generated data to a Parquet file for use in experiments

### `learning_curve_example/`

Analyzes learning curves and generates comparative plots:
- `src/main.rs`: Reads CSV data, applies Savitzky-Golay filtering, and calculates curve differences
- Generates plots comparing PolynomialLR and ExpHyperbolicLR learning curves

## Additional Scripts

### `bring.sh`

A simple shell script that copies the `hyperbolic_lr.py` file from the parent directory to the current directory.

### `scripts/01_poly_cos_lr.py`

Generates plots for Polynomial and Cosine Annealing learning rate schedulers:
- Implements `poly_lr` and `cos_lr` functions
- Creates plots for different epoch settings (250, 500, 750, 1000)

### `scripts/02_prop_1.py`

Generates a plot demonstrating the properties of the hyperbolic function used in HyperbolicLR:
- Plots the hyperbolic curve and its asymptote
- Adds custom labels and arrows for clear visualization

### `scripts/plot_hyperbolic_lr.py`

Creates plots for HyperbolicLR and ExpHyperbolicLR:
- Generates learning rate curves for different epoch settings
- Plots the initial gradient of the learning rate
- Creates a log-scale plot for ExpHyperbolicLR

## Requirements

See `requirements.txt` for the list of Python dependencies.

For Rust components, ensure you have Rust installed. Dependencies for Rust are managed via Cargo and are listed in the respective `Cargo.toml` files.

## Running Experiments

1. Install the required Python dependencies:
   ```sh
   # UV - Recommended
   uv venv
   uv pip sync requirements.txt
   source .venv/bin/activate

   # Or pip
   pip install -r requirements.txt
   ```

2. Data generation
   
   - Oscillation
     ```sh
     cd osc 
     cargo run --release
     ```

   - Integral
     ```sh
     cd integral
     cargo run --release
     ```

3. Run experiments:
   
   - CIFAR-10
     ```sh
     python main_cifar10.py
     ```

   - Oscillation
     ```sh
     python main_osc.py
     ```

   - Integral
     ```sh
     python main_integral.py
     ```

## Analysis

After running experiments:

1. Use the Rust-based analysis tools in the `analyze/` directory:

   ```sh
   cd analyze
   cargo run --release
   ```

   or for accuracy analysis:

   ```sh
   cargo run --release --bin acc
   ```

## Additional Notes

- The `bring.sh` script is used to copy the main HyperbolicLR implementation into the experiment directory.
- Rust components provide efficient data generation, analysis, and metric calculation, complementing the Python-based experimentation framework.
- The scripts in the `scripts/` directory generate various plots used in the paper to illustrate the properties of different learning rate schedulers.

For more detailed information on the implementation and usage of HyperbolicLR, refer to the main README in the root directory of this repository.
