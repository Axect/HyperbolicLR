"""
Exact SGD dynamics on a quadratic landscape: the direct loss-domain test of epoch insensitivity.
For f(theta) = 1/2 sum_i lambda_i theta_i^2 with per-mode gradient noise sigma_i^2 = c * lambda_i,
the expected loss under SGD is exactly computable per mode:
E[theta^2]_{t+1} = (1 - eta_t lambda)^2 E[theta^2]_t + eta_t^2 sigma^2.
No training, no sampling: every learning curve below is an exact expectation.

Two outputs:
1. Loss curves overlaying budgets N in {50, 100, 150, 200} per scheduler family at a fixed
   noise level, plus the SLCD decoupling table (same formula as paper/analyze).
2. A regime map: final exact loss vs noise coefficient c, showing that the optimal scheduler
   depends on the noise regime (constant wins when noise is negligible, WSD at moderate noise,
   smooth annealing at strong noise).
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

ETA0 = 0.8
ETAMIN = 8e-4
DETA = ETA0 - ETAMIN
U = 250.0
COOLDOWN_FRAC = 0.2

N_MODES = 200
LAMBDAS = 1.0 / np.arange(1, N_MODES + 1)
THETA0_SQ = np.ones(N_MODES)

BUDGETS = [50, 100, 150, 200]
MAIN_NOISE = 0.3
NOISE_SWEEP = np.geomspace(1e-4, 2.0, 25)


def hyperbolic(t, N):
    g = np.sqrt(max((U - t) ** 2 - (U - N) ** 2, 0.0)) / U
    g0 = np.sqrt(U ** 2 - (U - N) ** 2) / U
    return ETA0 + DETA * (g - g0)


def cosine(t, N):
    return ETAMIN + DETA * (1 + np.cos(np.pi * t / N)) / 2


def linear(t, N):
    return ETA0 + (ETAMIN - ETA0) * t / N


def wsd(t, N, c=COOLDOWN_FRAC):
    start = (1 - c) * N
    if t <= start:
        return ETA0
    return ETAMIN + DETA * (1 - np.sqrt((t - start) / (c * N)))


def constant(t, N):
    return ETA0


FAMILIES = {
    "Constant": constant,
    "Cosine": cosine,
    "Linear": linear,
    "WSD": wsd,
    "HyperbolicLR": hyperbolic,
}


def exact_loss_curve(f, N, noise_c):
    m = THETA0_SQ.copy()
    losses = np.empty(N)
    for k in range(N):
        eta = f(float(k), N)
        m = (1 - eta * LAMBDAS) ** 2 * m + eta ** 2 * noise_c * LAMBDAS
        losses[k] = 0.5 * np.sum(LAMBDAS * m)
    return losses


def slcd(l1, l2):
    T = min(len(l1), len(l2))
    return float(np.mean(np.abs(l1[:T] - l2[:T]) / (l1[:T] + l2[:T])))


def main():
    curves = {name: {N: exact_loss_curve(f, N, MAIN_NOISE) for N in BUDGETS}
              for name, f in FAMILIES.items()}

    with plt.style.context(["science", "nature"]):
        fig, axes = plt.subplots(1, len(FAMILIES), figsize=(2.1 * len(FAMILIES), 2.3),
                                 sharey=True)
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(BUDGETS)))
        for ax, (name, by_budget) in zip(axes, curves.items()):
            for N, color in zip(BUDGETS, colors):
                ax.plot(np.arange(1, N + 1), by_budget[N], color=color,
                        linewidth=1, label=f"$N{{=}}{N}$")
            ax.set_yscale("log")
            ax.set_xlabel("Step $t$")
            ax.set_title(name, fontsize=7)
        axes[0].set_ylabel(r"Exact expected loss")
        axes[-1].legend(fontsize=5, loc="upper right")
        plt.tight_layout()
        fig.savefig("../figs/exact_dynamics_loss_curves.png", dpi=600,
                    bbox_inches="tight")
        plt.close()
        print("Saved: figs/exact_dynamics_loss_curves.png")

        fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4), sharey=False)
        colors_f = plt.cm.viridis(np.linspace(0.05, 0.85, len(FAMILIES)))
        for ax, N in zip(axes, [50, 200]):
            for (name, f), color in zip(FAMILIES.items(), colors_f):
                finals = [exact_loss_curve(f, N, c)[-1] for c in NOISE_SWEEP]
                ls = "--" if name == "Constant" else "-"
                ax.plot(NOISE_SWEEP, finals, color=color, linewidth=1,
                        linestyle=ls, label=name)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"Noise coefficient $c$")
            ax.set_title(rf"Final loss, $N{{=}}{N}$", fontsize=7)
        axes[0].set_ylabel("Exact final loss")
        axes[1].legend(fontsize=5, loc="upper left")
        plt.tight_layout()
        fig.savefig("../figs/exact_dynamics_regime_map.png", dpi=600,
                    bbox_inches="tight")
        plt.close()
        print("Saved: figs/exact_dynamics_regime_map.png")

    print(f"\nSLCD on exact loss curves at noise c={MAIN_NOISE} (lower = more epoch-insensitive):")
    pairs = [(50, 100), (50, 200), (100, 200), (150, 200)]
    header = " ".join(f"{a}-{b:<8}" for a, b in pairs)
    print(f"{'Scheduler':<14} {header} mean")
    for name, by_budget in curves.items():
        vals = [slcd(by_budget[a], by_budget[b]) for a, b in pairs]
        row = " ".join(f"{v:<10.4f}" for v in vals)
        print(f"{name:<14} {row} {np.mean(vals):.4f}")

    print(f"\nFinal exact loss per budget at noise c={MAIN_NOISE}:")
    print(f"{'Scheduler':<14} " + " ".join(f"N={N:<10}" for N in BUDGETS))
    for name, by_budget in curves.items():
        row = " ".join(f"{by_budget[N][-1]:<12.3e}" for N in BUDGETS)
        print(f"{name:<14} {row}")

    print("\nRegime crossovers (final loss winner at N=200):")
    for c in [2e-4, 0.05, 0.15, 0.3, 0.6, 1.0]:
        finals = {name: exact_loss_curve(f, 200, c)[-1] for name, f in FAMILIES.items()}
        winner = min(finals, key=finals.get)
        print(f"  c={c:<8} winner: {winner:<14} ({finals[winner]:.3e})")


if __name__ == "__main__":
    main()
