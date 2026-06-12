"""
Theory figures for the prefix-invariance / anchoring tradeoff framework.
Panel (a): prefix deviation |eta_{N1} - eta_{N2}|(t) across scheduler families.
Panel (b): budget-sensitivity density dN(eta)(t), i.e. where each family places its sensitivity mass.
Panel (c): anchoring error A(N) of HyperbolicLR as a function of N/U.
Panel (d): sensitivity mass M(N) per family (conservation law totals).
All quantities are in units of delta_eta and use closed forms verified in THEORY_REFRAMING.md.
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

ETA0 = 1e-2
ETAMIN = 1e-6
DETA = ETA0 - ETAMIN
U = 250.0
COOLDOWN_FRAC = 0.2


def hyperbolic(t, N):
    g = np.sqrt(np.maximum((U - t) ** 2 - (U - N) ** 2, 0.0)) / U
    g0 = np.sqrt(U ** 2 - (U - N) ** 2) / U
    return ETA0 + DETA * (g - g0)


def cosine(t, N):
    return ETAMIN + DETA * (1 + np.cos(np.pi * t / N)) / 2


def linear(t, N):
    return ETA0 + (ETAMIN - ETA0) * t / N


def wsd(t, N, c=COOLDOWN_FRAC):
    t = np.asarray(t, dtype=float)
    start = (1 - c) * N
    out = np.full_like(t, ETA0)
    mask = t > start
    s = (t[mask] - start) / (c * N)
    out[mask] = ETAMIN + DETA * (1 - np.sqrt(s))
    return out


FAMILIES = {
    "Cosine": cosine,
    "Linear": linear,
    "WSD": wsd,
    "HyperbolicLR": hyperbolic,
}

LINESTYLES = {"Cosine": "-", "Linear": "--", "WSD": "-", "HyperbolicLR": "-"}


def sensitivity_density(f, t, N, h=1e-4):
    return (f(t, N + h) - f(t, N - h)) / (2 * h)


def mass(f, N, h=1e-3, n_points=20001):
    # Conservation law M(N) = B'(N) - eta_N(N); integrating eta avoids the
    # 1/sqrt singularity of the sensitivity density at moving cooldown boundaries.
    def B(NN):
        t = np.linspace(0, NN, n_points)
        return np.trapezoid(f(t, NN), t)
    return (B(N + h) - B(N - h)) / (2 * h) - f(np.array([N]), N)[0]


def main():
    N1, N2 = 100.0, 200.0
    ts = np.linspace(0, N1, 2001)

    with plt.style.context(["science", "nature"]):
        fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.6))
        colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(FAMILIES)))

        # (a) prefix deviation
        ax = axes[0, 0]
        for (name, f), color in zip(FAMILIES.items(), colors):
            dev = np.abs(f(ts, N1) - f(ts, N2)) / DETA
            ax.plot(ts / N1, dev, color=color, linewidth=1, label=name)
        ax.set_xlabel(r"$t / N_1$")
        ax.set_ylabel(r"$|\eta_{N_1}(t) - \eta_{N_2}(t)| / \Delta\eta$")
        ax.set_title(r"(a) Prefix deviation, $N_1{=}100 \to N_2{=}200$", fontsize=7)
        ax.legend(fontsize=5, loc="upper left")

        # (b) sensitivity density (mass placement)
        ax = axes[0, 1]
        N = 100.0
        td = np.linspace(0, N * (1 - 1e-6), 2001)
        for (name, f), color in zip(FAMILIES.items(), colors):
            dens = sensitivity_density(f, td, N) * N / DETA
            ax.plot(td / N, dens, color=color, linewidth=1, label=name)
        ax.set_xlabel(r"$t / N$")
        ax.set_ylabel(r"$N\,\partial_N \eta_N(t) / \Delta\eta$")
        ax.set_title(r"(b) Sensitivity mass placement, $N{=}100$", fontsize=7)
        ax.set_ylim(-0.1, 4.0)

        # (c) anchoring error of HyperbolicLR
        ax = axes[1, 0]
        rho = np.linspace(0.05, 1.0, 400)
        A = 1 - np.sqrt(1 - (1 - rho) ** 2)
        ax.plot(rho, A, color="black", linewidth=1)
        for n_mark in [50, 100, 150, 200]:
            r = n_mark / U
            ax.plot(r, 1 - np.sqrt(1 - (1 - r) ** 2), "o", markersize=3,
                    color="tab:red")
            ax.annotate(rf"$N{{=}}{n_mark}$", (r, 1 - np.sqrt(1 - (1 - r) ** 2)),
                        textcoords="offset points", xytext=(6, 4), fontsize=5)
        ax.set_xlabel(r"$\rho = N / U$")
        ax.set_ylabel(r"$A(N) / \Delta\eta$")
        ax.set_title(r"(c) Anchoring error (the price), $U{=}250$", fontsize=7)

        # (d) sensitivity mass totals
        ax = axes[1, 1]
        Ns = np.linspace(20, 245, 100)
        for (name, f), color in zip(FAMILIES.items(), colors):
            Ms = np.array([mass(f, N) for N in Ns]) / DETA
            ax.plot(Ns, Ms, color=color, linewidth=1, label=name,
                    linestyle=LINESTYLES[name])
        ax.legend(fontsize=5, loc="center right")
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"$M(N) / \Delta\eta$")
        ax.set_title(r"(d) Total sensitivity mass $M(N) = B'(N) - \eta_N(N)$", fontsize=7)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        fig.savefig("../figs/theory_tradeoff_overview.png", dpi=600,
                    bbox_inches="tight")
        plt.close()
        print("Saved: figs/theory_tradeoff_overview.png")

    print("\nMass totals M(N)/deta at N=100:")
    for name, f in FAMILIES.items():
        print(f"  {name:<14} {mass(f, 100.0) / DETA:.4f}")


if __name__ == "__main__":
    main()
