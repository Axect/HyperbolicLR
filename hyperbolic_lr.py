import math

class HyperbolicLR:
    """
    HyperbolicLR

    Args:
        optimizer: Optimizer
        upper_bound: Upper bound on various max_iters
        max_iter: Maximum number of iterations
        init_lr: Initial learning rate
        min_lr: Minimum learning rate
    """
    def __init__(self, optimizer, upper_bound=1000, max_iter=100, init_lr=1e-2, min_lr=1e-4):
        if upper_bound <= max_iter:
            raise ValueError("upper_bound must be greater than max_iter")
        self._optimizer = optimizer
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.iter = 0

    def step(self):
        """
        Step with the inner optimizer and update the learning rate
        """
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """
        Zero out the gradients with the inner optimizer
        """
        self._optimizer.zero_grad()

    def get_last_lr(self):
        """
        Get the last learning rate
        """
        return self._get_lr()

    def _get_lr(self):
        """
        Get the learning rate
        """
        delta_eta = self.init_lr - self.min_lr
        x = self.iter
        N = self.max_iter
        U = self.upper_bound
        return self.min_lr + delta_eta * math.sqrt((N - x) / U * (2 - (x + N) / U))

    def _update_learning_rate(self):
        """
        Update the learning rate
        """
        self.iter += 1
        lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
