"""
Decay Rate Analysis: Compares instantaneous decay rate d(eta)/dt across schedulers.
Visualizes WHY hyperbolic decay behaves differently from cosine, polynomial, etc.
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

def numerical_derivative(f, t, dt=0.5):
    return (f(t + dt) - f(t - dt)) / (2 * dt)


def main():
    N = 200
    t = np.linspace(1, N - 1, 500)

    schedulers = {
        "Polynomial": lambda t_: polynomial_lr(t_, N),
        "Cosine": lambda t_: cosine_lr(t_, N),
        "Linear": lambda t_: linear_lr(t_, N),
        "HyperbolicLR": lambda t_: hyperbolic_lr(t_, N),
        "ExpHyperbolicLR": lambda t_: exp_hyperbolic_lr(t_, N),
    }

    with plt.style.context(['science', 'nature']):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 4))

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        styles = ['-', '--', ':', '-', '--']

        for (name, f), color, style in zip(schedulers.items(), colors, styles):
            # LR curves
            lr_vals = f(t)
            ax1.plot(t, lr_vals, label=name, color=color, linestyle=style, linewidth=1.2)

            # Decay rate (d eta / dt)
            decay_rate = np.array([numerical_derivative(f, ti) for ti in t])
            ax2.plot(t, -decay_rate, label=name, color=color, linestyle=style, linewidth=1.2)

        ax1.set_ylabel('Learning Rate', fontsize=8)
        ax1.set_xlabel('Epoch', fontsize=8)
        ax1.legend(fontsize=6)
        ax1.autoscale(tight=True)

        ax2.set_ylabel(r'$-d\eta/dt$ (Decay Rate)', fontsize=8)
        ax2.set_xlabel('Epoch', fontsize=8)
        ax2.set_yscale('log')
        ax2.autoscale(tight=True)

        plt.tight_layout()
        fig.savefig('../figs/decay_rate_analysis.png', dpi=600, bbox_inches='tight')
        plt.close()
        print("Saved: figs/decay_rate_analysis.png")


if __name__ == "__main__":
    main()
