"""
Cumulative LR Budget Analysis: Compares integral(eta, 0, N) across schedulers
for different epoch counts. Demonstrates that HyperbolicLR has similar cumulative
budgets across different N, which explains its epoch-insensitivity.
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots


def polynomial_lr(t, N, power=0.5):
    return (1.0 - t / N) ** power

def cosine_lr(t, N, eta_min=1e-4):
    return eta_min + 0.5 * (1 - eta_min) * (1.0 + np.cos(np.pi * t / N))

def hyperbolic_lr(t, N, U=1000, infimum=1e-3):
    delta = 1.0 - infimum
    return 1.0 + delta * (
        np.sqrt((N - t) / U * (2 - (N + t) / U))
        - np.sqrt(N / U * (2 - N / U))
    )

def exp_hyperbolic_lr(t, N, U=1000, infimum=1e-3):
    ratio = 1.0 / infimum
    return ratio ** (
        np.sqrt((N - t) / U * (2 - (N + t) / U))
        - np.sqrt(N / U * (2 - N / U))
    )

def linear_lr(t, N, end=1e-4):
    return 1.0 + (end - 1.0) * t / N


def cumulative_budget(f, N, n_points=1000):
    t = np.linspace(0, N, n_points)
    lr_vals = np.array([f(ti, N) for ti in t])
    return np.trapz(lr_vals, t)


def normalized_cumulative_curve(f, N, n_points=1000):
    t = np.linspace(0, N, n_points)
    lr_vals = np.array([f(ti, N) for ti in t])
    cum = np.cumsum(lr_vals) * (N / n_points)
    total = cum[-1]
    if total > 0:
        cum /= total
    return np.linspace(0, 1, n_points), cum


def main():
    epoch_counts = [50, 100, 150, 200, 250, 500]

    schedulers = {
        "Polynomial": polynomial_lr,
        "Cosine": cosine_lr,
        "Linear": linear_lr,
        "HyperbolicLR": hyperbolic_lr,
        "ExpHyperbolicLR": exp_hyperbolic_lr,
    }

    with plt.style.context(['science', 'nature']):
        fig, axes = plt.subplots(1, len(schedulers), figsize=(2.5 * len(schedulers), 2.5),
                                 sharey=True)

        colors_epoch = plt.cm.viridis(np.linspace(0.2, 0.9, len(epoch_counts)))

        for ax, (name, f) in zip(axes, schedulers.items()):
            for N, color in zip(epoch_counts, colors_epoch):
                t_norm, cum_norm = normalized_cumulative_curve(f, N)
                ax.plot(t_norm, cum_norm, color=color, linewidth=1, label=f"N={N}")

            ax.set_title(name, fontsize=7)
            ax.set_xlabel('Training Progress', fontsize=7)
            if ax == axes[0]:
                ax.set_ylabel('Normalized Cumulative Budget', fontsize=7)

        axes[-1].legend(fontsize=5, loc='lower right')
        plt.tight_layout()
        fig.savefig('../figs/cumulative_lr_budget.png', dpi=600, bbox_inches='tight')
        plt.close()
        print("Saved: figs/cumulative_lr_budget.png")

    # Also print absolute budgets for reference
    print("\nAbsolute Cumulative LR Budgets:")
    print(f"{'Scheduler':<18} " + " ".join(f"N={N:<5}" for N in epoch_counts))
    print("-" * 70)
    for name, f in schedulers.items():
        budgets = [cumulative_budget(f, N) for N in epoch_counts]
        budgets_str = " ".join(f"{b:<7.2f}" for b in budgets)
        print(f"{name:<18} {budgets_str}")

    # Normalized by N=200 budget (to show relative stability)
    print("\nBudgets Relative to N=200:")
    ref_idx = epoch_counts.index(200)
    for name, f in schedulers.items():
        budgets = [cumulative_budget(f, N) for N in epoch_counts]
        ref = budgets[ref_idx]
        ratios_str = " ".join(f"{b/ref:<7.3f}" for b in budgets)
        print(f"{name:<18} {ratios_str}")


if __name__ == "__main__":
    main()
