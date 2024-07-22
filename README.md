# Hyperbolic Learning rate Scheduler

This repository contains the implementation and experimental code for the paper "_HyperbolicLR: Epoch Insensitive Learning Rate Scheduler_".
It includes the novel learning rate schedulers, HyperbolicLR and ExpHyperbolicLR, designed to address the learning curve decoupling problem in deep learning.

## Overview

HyperbolicLR and ExpHyperbolicLR are learning rate schedulers that maintain consistent initial learning rate changes, regardless of the total number of epochs.
This property helps mitigate the learning curve decoupling problem observed in conventional schedulers, potentially leading to more robust and efficient training of deep neural networks.

## Repository Structure

- `hyperbolic_lr.py`: Implementation of HyperbolicLR and ExpHyperbolicLR schedulers.
- `paper/`: Directory containing all experimental code, data, and results used in the associated research paper.
  - `experiments/`: Experimental setup and execution scripts.
  - `figs/`: Figures generated for the paper.
  - `results/`: Analysis of experimental results.
  - `scripts/`: Utility scripts for data processing and visualization.

## Installation

To use the HyperbolicLR and ExpHyperbolicLR schedulers in your project:

1. Clone this repository:
   ```
   git clone https://github.com/Axect/HyperbolicLR
   ```

2. Copy `hyperbolic_lr.py` to your project directory or add this repository to your Python path.

## Usage

Here's a basic example of how to use HyperbolicLR in your PyTorch project:

```python
from hyperbolic_lr import HyperbolicLR # or ExpHyperbolicLR
import torch

# Define your model and optimizer
model = YourModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
num_epochs = 100

# Create the HyperbolicLR scheduler
scheduler = HyperbolicLR(optimizer, upper_bound=250, max_iter=num_epochs, init_lr=1e-2, infimum_lr=1e-5)

# Training loop
for epoch in range(num_epochs):
    train(model, optimizer)
    scheduler.step()
```

For more detailed usage and examples, please refer to the paper and experimental code in the `paper/` directory.

## Citing

If you use HyperbolicLR or ExpHyperbolicLR in your research, please cite our paper:

```
[Citation information will be added upon publication]
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

We welcome contributions to improve HyperbolicLR and ExpHyperbolicLR. Please feel free to submit issues or pull requests.

## Contact

For any questions or discussions regarding this project, please open an issue in this repository.
