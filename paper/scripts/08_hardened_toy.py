"""
Hardened exact-dynamics study: every scheduler gets its hyperparameters tuned per
(noise, budget) before any comparison is made, removing the untuned-baseline objection.

Parts:
A. Tuned regime map: per-family best final loss vs noise coefficient, N in {50, 200}.
   Tuning grids: eta0 for everyone; cooldown fraction for WSD; U for HyperbolicLR.
B. HPO transfer (the paper's primary claim, in exact dynamics): tune at N=50, deploy the
   same hyperparameters at N=200 (schedule re-instantiated with the new horizon),
   report regret vs tuning directly at N=200.
C. Spectrum robustness: repeat the tuned comparison for lambda_i = i^{-alpha},
   alpha in {0.5, 1, 2}.

All learning curves are exact expectations of SGD on quadratics with per-mode noise
sigma_i^2 = c * lambda_i (recursion m <- (1 - eta lam)^2 m + eta^2 c lam).
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

N_MODES = 200
ETA_RATIO = 1e-3          # eta_min = ETA_RATIO * eta0 for all annealing schedules
ETA0_GRID = np.geomspace(0.05, 1.6, 10)
WSD_C_GRID = [0.1, 0.2, 0.3, 0.5, 0.7]
HYP_U_GRID = [210.0, 250.0, 300.0, 400.0, 600.0]


def spectrum(alpha):
    return 1.0 / np.arange(1, N_MODES + 1) ** alpha


def etas_constant(N, eta0):
    return np.full(N, eta0)


def etas_cosine(N, eta0):
    t = np.arange(N)
    emin = ETA_RATIO * eta0
    return emin + (eta0 - emin) * (1 + np.cos(np.pi * t / N)) / 2


def etas_linear(N, eta0):
    t = np.arange(N)
    emin = ETA_RATIO * eta0
    return eta0 + (emin - eta0) * t / N


def etas_wsd(N, eta0, c):
    t = np.arange(N, dtype=float)
    emin = ETA_RATIO * eta0
    start = (1 - c) * N
    out = np.full(N, eta0)
    mask = t > start
    s = (t[mask] - start) / (c * N)
    out[mask] = emin + (eta0 - emin) * (1 - np.sqrt(s))
    return out


def etas_hyperbolic(N, eta0, Uval):
    t = np.arange(N, dtype=float)
    emin = ETA_RATIO * eta0
    g = np.sqrt(np.maximum((Uval - t) ** 2 - (Uval - N) ** 2, 0.0)) / Uval
    g0 = np.sqrt(Uval ** 2 - (Uval - N) ** 2) / Uval
    return eta0 + (eta0 - emin) * (g - g0)


def param_grid(family, N):
    if family == "Constant":
        return [(e,) for e in ETA0_GRID]
    if family in ("Cosine", "Linear"):
        return [(e,) for e in ETA0_GRID]
    if family == "WSD":
        return [(e, c) for e in ETA0_GRID for c in WSD_C_GRID]
    if family == "HyperbolicLR":
        return [(e, u) for e in ETA0_GRID for u in HYP_U_GRID if u >= N]
    raise ValueError(family)


def make_etas(family, N, params):
    if family == "Constant":
        return etas_constant(N, *params)
    if family == "Cosine":
        return etas_cosine(N, *params)
    if family == "Linear":
        return etas_linear(N, *params)
    if family == "WSD":
        return etas_wsd(N, *params)
    if family == "HyperbolicLR":
        return etas_hyperbolic(N, *params)
    raise ValueError(family)


FAMILIES = ["Constant", "Cosine", "Linear", "WSD", "HyperbolicLR"]


def final_loss(etas, noise_c, lams):
    m = np.ones(N_MODES)
    for eta in etas:
        decay = 1 - eta * lams
        m = decay * decay * m + eta * eta * noise_c * lams
    return 0.5 * float(np.sum(lams * m))


def tune(family, N, noise_c, lams):
    best, best_params = np.inf, None
    for params in param_grid(family, N):
        loss = final_loss(make_etas(family, N, params), noise_c, lams)
        if loss < best:
            best, best_params = loss, params
    return best, best_params


def part_a_tuned_regime_map():
    lams = spectrum(1.0)
    noises = np.geomspace(1e-4, 2.0, 13)
    results = {N: {fam: [] for fam in FAMILIES} for N in (50, 200)}
    for N in (50, 200):
        for c in noises:
            for fam in FAMILIES:
                best, _ = tune(fam, N, c, lams)
                results[N][fam].append(best)

    with plt.style.context(["science", "nature"]):
        fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4))
        colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(FAMILIES)))
        for ax, N in zip(axes, (50, 200)):
            for fam, color in zip(FAMILIES, colors):
                ls = "--" if fam == "Constant" else "-"
                ax.plot(noises, results[N][fam], color=color, linewidth=1,
                        linestyle=ls, label=fam)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"Noise coefficient $c$")
            ax.set_title(rf"Tuned final loss, $N{{=}}{N}$", fontsize=7)
        axes[0].set_ylabel("Exact final loss (tuned)")
        axes[1].legend(fontsize=5, loc="upper left")
        plt.tight_layout()
        fig.savefig("../figs/tuned_regime_map.png", dpi=600, bbox_inches="tight")
        plt.close()
    print("Saved: figs/tuned_regime_map.png")

    print("\nPart A: tuned winners (final loss):")
    for N in (50, 200):
        print(f"  N={N}:")
        for c_idx, c in enumerate(noises):
            losses = {fam: results[N][fam][c_idx] for fam in FAMILIES}
            ranked = sorted(losses, key=lambda f: losses[f])
            margin = losses[ranked[1]] / losses[ranked[0]] - 1
            print(f"    c={c:9.4f}  winner={ranked[0]:<13} (runner-up {ranked[1]}, margin {100*margin:.1f}%)")
    return results


def part_b_hpo_transfer():
    lams = spectrum(1.0)
    print("\nPart B: HPO transfer, tune at N=50 then deploy at N=200 (schedule re-instantiated).")
    print("Regret = transferred_loss / tuned_at_200_loss - 1 (lower is better).")
    print(f"{'noise c':<9} " + " ".join(f"{fam:<14}" for fam in FAMILIES))
    for c in [0.05, 0.3, 1.0]:
        regrets = []
        for fam in FAMILIES:
            _, params50 = tune(fam, 50, c, lams)
            if fam == "HyperbolicLR" and params50[1] < 200:
                params200 = (params50[0], max(params50[1], 210.0))
            else:
                params200 = params50
            transferred = final_loss(make_etas(fam, 200, params200), c, lams)
            tuned200, _ = tune(fam, 200, c, lams)
            regrets.append(transferred / tuned200 - 1)
        print(f"{c:<9} " + " ".join(f"{100*r:<14.2f}" for r in regrets))
    print("(values in %; HyperbolicLR U clipped up to >= N when transferring)")


def part_c_spectrum_robustness():
    print("\nPart C: spectrum robustness, tuned winners at N=200:")
    print(f"{'alpha':<7} {'c=0.05':<28} {'c=0.3':<28} {'c=1.0':<28}")
    for alpha in [0.5, 1.0, 2.0]:
        lams = spectrum(alpha)
        cells = []
        for c in [0.05, 0.3, 1.0]:
            losses = {fam: tune(fam, 200, c, lams)[0] for fam in FAMILIES}
            ranked = sorted(losses, key=lambda f: losses[f])
            margin = losses[ranked[1]] / losses[ranked[0]] - 1
            cells.append(f"{ranked[0]} (+{100*margin:.1f}%)")
        print(f"{alpha:<7} {cells[0]:<28} {cells[1]:<28} {cells[2]:<28}")


if __name__ == "__main__":
    part_a_tuned_regime_map()
    part_b_hpo_transfer()
    part_c_spectrum_robustness()
