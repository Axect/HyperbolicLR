import math

class HyperbolicLR:
    """
    HyperbolicLR

    Args:
        optimizer: Optimizer
        upper_bound: Upper bound on various max_iters
        max_iter: Maximum number of iterations
        infimum_lr: The infimum of the hyperbolic learning rate
        warmup_epochs: Number of warmup epochs (linear ramp from infimum_lr to init_lr)
    """
    def __init__(self, optimizer, upper_bound=1000, max_iter=100, infimum_lr=1e-6,
                 warmup_epochs=0):
        init_lr = optimizer.param_groups[0]['lr']
        if upper_bound < max_iter:
            raise ValueError("upper_bound must be greater than max_iter")
        elif infimum_lr >= init_lr:
            raise ValueError("infimum_lr must be less than init_lr")
        self._optimizer = optimizer
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.init_lr = init_lr
        self.infimum_lr = infimum_lr
        self.delta_lr = init_lr - infimum_lr
        self.warmup_epochs = warmup_epochs
        self.decay_iter = max_iter - warmup_epochs
        self.iter = 0

    def step(self):
        self._update_learning_rate()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self._optimizer.param_groups]

    def _get_lr(self):
        x = self.iter
        if x <= self.warmup_epochs and self.warmup_epochs > 0:
            return self.infimum_lr + (self.init_lr - self.infimum_lr) * x / self.warmup_epochs

        x_decay = x - self.warmup_epochs
        N = self.decay_iter
        U = self.upper_bound
        return self.init_lr + self.delta_lr * (
            math.sqrt((N - x_decay) / U * (2 - (N + x_decay) / U)) - math.sqrt(N / U * (2 - N / U))
        )

    def _update_learning_rate(self):
        self.iter += 1
        lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class ExpHyperbolicLR:
    """
    ExpHyperbolicLR

    Args:
        optimizer: Optimizer
        upper_bound: Upper bound on various max_iters
        max_iter: Maximum number of iterations
        infimum_lr: The infimum of the hyperbolic learning rate
        warmup_epochs: Number of warmup epochs (linear ramp from infimum_lr to init_lr)
    """
    def __init__(self, optimizer, upper_bound=1000, max_iter=100, infimum_lr=1e-6,
                 warmup_epochs=0):
        init_lr = optimizer.param_groups[0]['lr']
        if upper_bound < max_iter:
            raise ValueError("upper_bound must be greater than max_iter")
        elif infimum_lr >= init_lr:
            raise ValueError("infimum_lr must be less than init_lr")
        self._optimizer = optimizer
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.init_lr = init_lr
        self.infimum_lr = infimum_lr
        self.lr_ratio = init_lr / infimum_lr
        self.warmup_epochs = warmup_epochs
        self.decay_iter = max_iter - warmup_epochs
        self.iter = 0

    def step(self):
        self._update_learning_rate()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self._optimizer.param_groups]

    def _get_lr(self):
        x = self.iter
        if x <= self.warmup_epochs and self.warmup_epochs > 0:
            return self.infimum_lr + (self.init_lr - self.infimum_lr) * x / self.warmup_epochs

        x_decay = x - self.warmup_epochs
        N = self.decay_iter
        U = self.upper_bound
        return self.init_lr * self.lr_ratio ** (
            math.sqrt((N - x_decay) / U * (2 - (N + x_decay) / U)) - math.sqrt(N / U * (2 - N / U))
        )

    def _update_learning_rate(self):
        self.iter += 1
        lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
