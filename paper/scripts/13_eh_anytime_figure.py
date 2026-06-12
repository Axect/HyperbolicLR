"""
Figure for the ExpHyperbolicLR promotion (THEORY_REFRAMING.md 7.3):
left  - schedule shapes on log scale: the dense-game free-form optimum is near-geometric,
        and tuned EH tracks it closely while tuned H bends away;
right - dense anytime regret curves: EH roughly halves cosine/H regret across the window.
Reuses the cached dense oracle and the saved free-form optimum from script 11.
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

N_MODES = 200
LAMS = 1.0 / np.arange(1, N_MODES + 1)
ETA_RATIO = 1e-3
TAU_MIN, TAU_MAX = 50, 200
TAUS = np.arange(TAU_MIN, TAU_MAX + 1)
NOISE_C = 0.3

ORACLE = dict(zip(*[a.tolist() for a in
                    np.load("../figs/dense_oracle_c0.3.npz").values()]))
FF_ETAS = np.load("../figs/dense_anytime_freeform.npz")["etas"]

ETA0_GRID = np.geomspace(0.05, 1.9, 12)


def loss_trajectory(etas):
    m = np.ones(N_MODES)
    out = np.empty(len(etas))
    for k, eta in enumerate(etas):
        d = 1 - eta * LAMS
        m = d * d * m + eta * eta * NOISE_C * LAMS
        out[k] = 0.5 * float(np.sum(LAMS * m))
    return out


def regrets(etas):
    traj = loss_trajectory(etas)
    return np.array([traj[t - 1] / ORACLE[t] - 1 for t in TAUS])


def hold_tail(e, total):
    if len(e) >= total:
        return e[:total]
    return np.concatenate([e, np.full(total - len(e), e[-1])])


def sched_cosine(total, eta0, nplan):
    t = np.arange(nplan)
    em = ETA_RATIO * eta0
    return hold_tail(em + (eta0 - em) * (1 + np.cos(np.pi * t / nplan)) / 2, total)


def sched_hyperbolic(total, eta0, nplan, ufac):
    Uv = ufac * nplan
    t = np.arange(nplan, dtype=float)
    em = ETA_RATIO * eta0
    g = np.sqrt(np.maximum((Uv - t) ** 2 - (Uv - nplan) ** 2, 0.0)) / Uv
    g0 = np.sqrt(Uv ** 2 - (Uv - nplan) ** 2) / Uv
    return hold_tail(eta0 + (eta0 - em) * (g - g0), total)


def sched_exphyperbolic(total, eta0, nplan, ufac):
    Uv = ufac * nplan
    t = np.arange(nplan, dtype=float)
    em = ETA_RATIO * eta0
    g = np.sqrt(np.maximum((Uv - t) ** 2 - (Uv - nplan) ** 2, 0.0)) / Uv
    g0 = np.sqrt(Uv ** 2 - (Uv - nplan) ** 2) / Uv
    return hold_tail(eta0 * (eta0 / em) ** (g - g0), total)


def tune(builder, grids):
    best = (np.inf, None)
    for params in grids:
        mm = float(regrets(builder(TAU_MAX, *params)).max())
        if mm < best[0]:
            best = (mm, params)
    return best[1]


def main():
    cfg_eh = tune(sched_exphyperbolic,
                  [(e, n, u) for e in ETA0_GRID for n in [100, 125, 150, 175, 200, 250, 300]
                   for u in [1.02, 1.05, 1.1, 1.2, 1.5]])
    cfg_h = tune(sched_hyperbolic,
                 [(e, n, u) for e in ETA0_GRID for n in [50, 75, 100, 125, 150, 175, 200]
                  for u in [1.05, 1.2, 1.5, 2.5]])
    cfg_cos = tune(sched_cosine,
                   [(e, n) for e in ETA0_GRID for n in [50, 75, 100, 125, 150, 175, 200]])
    print(f"tuned EH {cfg_eh}, H {cfg_h}, cosine {cfg_cos}")

    eh = sched_exphyperbolic(TAU_MAX, *cfg_eh)
    h = sched_hyperbolic(TAU_MAX, *cfg_h)
    cos = sched_cosine(TAU_MAX, *cfg_cos)

    with plt.style.context(["science", "nature"]):
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5))
        st = np.arange(TAU_MAX)

        ax = axes[0]
        ax.plot(st, FF_ETAS, color="tab:red", linewidth=1.0,
                label="Free-form optimum")
        ax.plot(st, eh, color="tab:purple", linewidth=1.0, linestyle="--",
                label="ExpHyperbolicLR (tuned)")
        ax.plot(st, h, color="tab:green", linewidth=0.8, linestyle="-.",
                label="HyperbolicLR (tuned)")
        ax.plot(st, cos, color="tab:blue", linewidth=0.8, linestyle=":",
                label="Cosine (tuned)")
        ax.set_yscale("log")
        ax.set_xlabel("Step $t$")
        ax.set_ylabel(r"$\eta_t$")
        ax.set_title("Committed schedules vs free-form optimum", fontsize=7)
        ax.legend(fontsize=5, loc="lower left")

        ax = axes[1]
        for etas, name, color, ls in [
            (FF_ETAS, "Free-form", "tab:red", "-"),
            (eh, "ExpHyperbolicLR", "tab:purple", "--"),
            (h, "HyperbolicLR", "tab:green", "-."),
            (cos, "Cosine", "tab:blue", ":"),
        ]:
            r = regrets(etas)
            ax.plot(TAUS, 100 * r, color=color, linewidth=0.9, linestyle=ls,
                    label=f"{name} (max {100*r.max():.0f}\\%, mean {100*r.mean():.0f}\\%)")
        ax.set_yscale("log")
        ax.set_xlabel(r"Stop time $\tau$")
        ax.set_ylabel(r"Anytime regret [\%]")
        ax.set_title("Dense anytime regret", fontsize=7)
        ax.legend(fontsize=5, loc="upper center")

        plt.tight_layout()
        fig.savefig("../figs/eh_anytime.png", dpi=600, bbox_inches="tight")
        plt.close()
    print("Saved: figs/eh_anytime.png")


if __name__ == "__main__":
    main()
