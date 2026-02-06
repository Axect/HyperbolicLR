import numpy as np
import matplotlib.pyplot as plt
import scienceplots

class PolynomialLR:
    def __init__(self, init_lr, max_epoch, power):
        self.init_lr = init_lr
        self.max_epoch = max_epoch
        self.power = power

    def get_lr(self, epoch):
        return self.init_lr * (1.0 - epoch / self.max_epoch) ** self.power

class CosineAnnealingLR:
    def __init__(self, min_lr, max_lr, max_epoch):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.max_epoch = max_epoch
        self.init_lr = max_lr  # Added this line

    def get_lr(self, epoch):
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + np.cos(np.pi * epoch / self.max_epoch))

class HyperbolicLR:
    def __init__(self, init_lr, infimum_lr, max_epoch, upper_bound):
        self.init_lr = init_lr
        self.infimum_lr = infimum_lr
        self.max_epoch = max_epoch
        self.upper_bound = upper_bound

    def get_lr(self, epoch):
        delta_lr = self.init_lr - self.infimum_lr
        N = self.max_epoch
        U = self.upper_bound
        return self.init_lr + delta_lr * (
            np.sqrt((N - epoch) / U * (2 - (N + epoch) / U))
            - np.sqrt(N / U * (2 - N / U))
        )

class ExpHyperbolicLR:
    def __init__(self, init_lr, infimum_lr, max_epoch, upper_bound):
        self.init_lr = init_lr
        self.infimum_lr = infimum_lr
        self.max_epoch = max_epoch
        self.upper_bound = upper_bound

    def get_lr(self, epoch):
        delta_lr = self.init_lr / self.infimum_lr
        N = self.max_epoch
        U = self.upper_bound
        return self.init_lr * delta_lr ** (
            np.sqrt((N - epoch) / U * (2 - (N + epoch) / U))
            - np.sqrt(N / U * (2 - N / U))
        )

class LinearLR:
    def __init__(self, init_lr, end_lr, max_epoch):
        self.init_lr = init_lr
        self.end_lr = end_lr
        self.max_epoch = max_epoch

    def get_lr(self, epoch):
        return self.init_lr + (self.end_lr - self.init_lr) * epoch / self.max_epoch

class StepLR:
    def __init__(self, init_lr, step_size, gamma):
        self.init_lr = init_lr
        self.step_size = step_size
        self.gamma = gamma
        self.max_epoch = step_size * 3  # for plotting

    def get_lr(self, epoch):
        return self.init_lr * self.gamma ** (epoch // self.step_size)

class OneCycleLR:
    def __init__(self, max_lr, max_epoch, pct_start=0.3):
        self.max_lr = max_lr
        self.max_epoch = max_epoch
        self.pct_start = pct_start
        self.init_lr = max_lr

    def get_lr(self, epoch):
        if epoch < self.max_epoch * self.pct_start:
            return self.max_lr * epoch / (self.max_epoch * self.pct_start)
        else:
            progress = (epoch - self.max_epoch * self.pct_start) / (self.max_epoch * (1 - self.pct_start))
            return self.max_lr * 0.5 * (1 + np.cos(np.pi * progress))

class CyclicLR:
    def __init__(self, base_lr, max_lr, step_size_up, max_epoch):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.max_epoch = max_epoch
        self.init_lr = max_lr

    def get_lr(self, epoch):
        cycle = np.floor(1 + epoch / (2 * self.step_size_up))
        x = abs(epoch / self.step_size_up - 2 * cycle + 1)
        return self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x)

class WarmupHyperbolicLR:
    def __init__(self, init_lr, infimum_lr, max_epoch, upper_bound, warmup_epochs):
        self.init_lr = init_lr
        self.infimum_lr = infimum_lr
        self.max_epoch = max_epoch
        self.upper_bound = upper_bound
        self.warmup_epochs = warmup_epochs

    def get_lr(self, epoch):
        if epoch <= self.warmup_epochs:
            return self.infimum_lr + (self.init_lr - self.infimum_lr) * epoch / self.warmup_epochs
        x = epoch - self.warmup_epochs
        N = self.max_epoch - self.warmup_epochs
        U = self.upper_bound
        delta_lr = self.init_lr - self.infimum_lr
        return self.init_lr + delta_lr * (
            np.sqrt((N - x) / U * (2 - (N + x) / U))
            - np.sqrt(N / U * (2 - N / U))
        )

class WarmupCosineAnnealingLR:
    def __init__(self, min_lr, max_lr, max_epoch, warmup_epochs):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.max_epoch = max_epoch
        self.warmup_epochs = warmup_epochs
        self.init_lr = max_lr

    def get_lr(self, epoch):
        if epoch <= self.warmup_epochs:
            return self.min_lr + (self.max_lr - self.min_lr) * epoch / self.warmup_epochs
        x = epoch - self.warmup_epochs
        T = self.max_epoch - self.warmup_epochs
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + np.cos(np.pi * x / T))


def plot_lrs(lrs, name):
    with plt.style.context(['science', 'nature']):
        fig, ax = plt.subplots()

        legends = [r"$N=250$", r"$N=500$", r"$N=750$", r"$N=1000$"]
        line_styles = ['-', '--', ':', '-.']
        colors = ['darkblue', 'red', 'darkgreen', 'black']

        for lr, legend, line_style, color in zip(lrs, legends, line_styles, colors):
            epochs = np.linspace(0, lr.max_epoch, 1000)
            learning_rates = [lr.get_lr(epoch) for epoch in epochs]
            ax.plot(epochs, learning_rates, label=legend, linestyle=line_style, color=color, linewidth=1.25)

        ax.autoscale(tight=True)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('Learning Rate', fontsize=8)
        ax.legend(fontsize=8)

        if 'exp' in name:
            ax.set_yscale('log')

        plt.tight_layout()
        fig.savefig(f'{name}.png', dpi=600, bbox_inches='tight')
        plt.close()

def main():
    poly_lrs = [
        PolynomialLR(init_lr=1, max_epoch=250, power=0.5),
        PolynomialLR(init_lr=1, max_epoch=500, power=0.5),
        PolynomialLR(init_lr=1, max_epoch=750, power=0.5),
        PolynomialLR(init_lr=1, max_epoch=1000, power=0.5),
    ]

    cos_lrs = [
        CosineAnnealingLR(min_lr=1e-4, max_lr=1, max_epoch=250),
        CosineAnnealingLR(min_lr=1e-4, max_lr=1, max_epoch=500),
        CosineAnnealingLR(min_lr=1e-4, max_lr=1, max_epoch=750),
        CosineAnnealingLR(min_lr=1e-4, max_lr=1, max_epoch=1000),
    ]

    hyp_lrs = [
        HyperbolicLR(init_lr=1, infimum_lr=1e-3, max_epoch=250, upper_bound=1000),
        HyperbolicLR(init_lr=1, infimum_lr=1e-3, max_epoch=500, upper_bound=1000),
        HyperbolicLR(init_lr=1, infimum_lr=1e-3, max_epoch=750, upper_bound=1000),
        HyperbolicLR(init_lr=1, infimum_lr=1e-3, max_epoch=1000, upper_bound=1000),
    ]

    exp_hyp_lrs = [
        ExpHyperbolicLR(init_lr=1, infimum_lr=1e-3, max_epoch=250, upper_bound=1000),
        ExpHyperbolicLR(init_lr=1, infimum_lr=1e-3, max_epoch=500, upper_bound=1000),
        ExpHyperbolicLR(init_lr=1, infimum_lr=1e-3, max_epoch=750, upper_bound=1000),
        ExpHyperbolicLR(init_lr=1, infimum_lr=1e-3, max_epoch=1000, upper_bound=1000),
    ]

    linear_lrs = [
        LinearLR(init_lr=1, end_lr=1e-4, max_epoch=250),
        LinearLR(init_lr=1, end_lr=1e-4, max_epoch=500),
        LinearLR(init_lr=1, end_lr=1e-4, max_epoch=750),
        LinearLR(init_lr=1, end_lr=1e-4, max_epoch=1000),
    ]

    step_lrs = [
        StepLR(init_lr=1, step_size=83, gamma=0.1),
        StepLR(init_lr=1, step_size=166, gamma=0.1),
        StepLR(init_lr=1, step_size=250, gamma=0.1),
        StepLR(init_lr=1, step_size=333, gamma=0.1),
    ]

    onecycle_lrs = [
        OneCycleLR(max_lr=1, max_epoch=250, pct_start=0.3),
        OneCycleLR(max_lr=1, max_epoch=500, pct_start=0.3),
        OneCycleLR(max_lr=1, max_epoch=750, pct_start=0.3),
        OneCycleLR(max_lr=1, max_epoch=1000, pct_start=0.3),
    ]

    warmup_hyp_lrs = [
        WarmupHyperbolicLR(init_lr=1, infimum_lr=1e-3, max_epoch=250, upper_bound=1000, warmup_epochs=25),
        WarmupHyperbolicLR(init_lr=1, infimum_lr=1e-3, max_epoch=500, upper_bound=1000, warmup_epochs=50),
        WarmupHyperbolicLR(init_lr=1, infimum_lr=1e-3, max_epoch=750, upper_bound=1000, warmup_epochs=75),
        WarmupHyperbolicLR(init_lr=1, infimum_lr=1e-3, max_epoch=1000, upper_bound=1000, warmup_epochs=100),
    ]

    warmup_cos_lrs = [
        WarmupCosineAnnealingLR(min_lr=1e-4, max_lr=1, max_epoch=250, warmup_epochs=25),
        WarmupCosineAnnealingLR(min_lr=1e-4, max_lr=1, max_epoch=500, warmup_epochs=50),
        WarmupCosineAnnealingLR(min_lr=1e-4, max_lr=1, max_epoch=750, warmup_epochs=75),
        WarmupCosineAnnealingLR(min_lr=1e-4, max_lr=1, max_epoch=1000, warmup_epochs=100),
    ]

    plot_lrs(poly_lrs, "figs/poly")
    plot_lrs(cos_lrs, "figs/cos")
    plot_lrs(hyp_lrs, "figs/hyp")
    plot_lrs(exp_hyp_lrs, "figs/exp_hyp")
    plot_lrs(linear_lrs, "figs/linear")
    plot_lrs(step_lrs, "figs/step")
    plot_lrs(onecycle_lrs, "figs/onecycle")
    plot_lrs(warmup_hyp_lrs, "figs/warmup_hyp")
    plot_lrs(warmup_cos_lrs, "figs/warmup_cos")

if __name__ == "__main__":
    main()
