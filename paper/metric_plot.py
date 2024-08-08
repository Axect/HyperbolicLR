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

    plot_lrs(poly_lrs, "figs/poly")
    plot_lrs(cos_lrs, "figs/cos")
    plot_lrs(hyp_lrs, "figs/hyp")
    plot_lrs(exp_hyp_lrs, "figs/exp_hyp")

if __name__ == "__main__":
    main()
