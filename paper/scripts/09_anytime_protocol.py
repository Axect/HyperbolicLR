"""
Anytime protocol on exact dynamics: quantify how gracefully each scheduler degrades when
training must stop at an unknown time tau.

Game:
- tau is drawn from [TAU_MIN, TAU_MAX]; the contender does NOT know tau in advance.
- Each contender commits to one configuration (eta0, horizon guess N_plan, family-specific
  params) chosen to minimize its WORST-CASE regret over tau (minimax; fair tuning).
- After t >= N_plan the schedule holds its last LR (PyTorch convention).
- Oracle(tau) = best final loss over ALL families tuned with tau known in advance.
- Regret R(tau) = L_contender(tau) / Oracle(tau) - 1.
- WSD enters twice: "WSD-stop" (just stop mid-run) and "WSD-branch" (on the stop request,
  run an extra cooldown of length BRANCH_FRAC * tau from the checkpoint; costs extra compute,
  reported both ignoring and charging that overhead).

All dynamics are exact expectations of SGD on the quadratic toy (see 07/08 scripts).
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

N_MODES = 200
LAMS = 1.0 / np.arange(1, N_MODES + 1)
ETA_RATIO = 1e-3
TAU_MIN, TAU_MAX = 50, 200
TAUS = np.arange(TAU_MIN, TAU_MAX + 1, 10)
BRANCH_FRAC = 0.2

ETA0_GRID = np.geomspace(0.05, 1.9, 12)
NPLAN_GRID = [50, 75, 100, 125, 150, 175, 200]
U_FACTORS = [1.05, 1.2, 1.5, 2.5]
WSD_CD_GRID = [0.2, 0.5]


def hold_tail(etas, total):
    if len(etas) >= total:
        return etas[:total]
    return np.concatenate([etas, np.full(total - len(etas), etas[-1])])


def sched_constant(total, eta0):
    return np.full(total, eta0)


def sched_cosine(total, eta0, nplan):
    t = np.arange(nplan)
    emin = ETA_RATIO * eta0
    e = emin + (eta0 - emin) * (1 + np.cos(np.pi * t / nplan)) / 2
    return hold_tail(e, total)


def sched_linear(total, eta0, nplan):
    t = np.arange(nplan)
    emin = ETA_RATIO * eta0
    e = eta0 + (emin - eta0) * t / nplan
    return hold_tail(e, total)


def sched_wsd(total, eta0, nplan, cd):
    t = np.arange(nplan, dtype=float)
    emin = ETA_RATIO * eta0
    start = (1 - cd) * nplan
    e = np.full(nplan, eta0)
    mask = t > start
    s = (t[mask] - start) / (cd * nplan)
    e[mask] = emin + (eta0 - emin) * (1 - np.sqrt(s))
    return hold_tail(e, total)


def sched_hyperbolic(total, eta0, nplan, ufac):
    Uval = ufac * nplan
    t = np.arange(nplan, dtype=float)
    emin = ETA_RATIO * eta0
    g = np.sqrt(np.maximum((Uval - t) ** 2 - (Uval - nplan) ** 2, 0.0)) / Uval
    g0 = np.sqrt(Uval ** 2 - (Uval - nplan) ** 2) / Uval
    e = eta0 + (eta0 - emin) * (g - g0)
    return hold_tail(e, total)


def loss_trajectory(etas, noise_c):
    m = np.ones(N_MODES)
    out = np.empty(len(etas))
    for k, eta in enumerate(etas):
        d = 1 - eta * LAMS
        m = d * d * m + eta * eta * noise_c * LAMS
        out[k] = 0.5 * float(np.sum(LAMS * m))
    return out


def state_at(etas, steps, noise_c):
    m = np.ones(N_MODES)
    for eta in etas[:steps]:
        d = 1 - eta * LAMS
        m = d * d * m + eta * eta * noise_c * LAMS
    return m


def branch_cooldown_loss(m, eta_start, length, noise_c):
    emin = ETA_RATIO * eta_start
    s = (np.arange(length) + 1) / length
    etas = emin + (eta_start - emin) * (1 - np.sqrt(s))
    for eta in etas:
        d = 1 - eta * LAMS
        m = d * d * m + eta * eta * noise_c * LAMS
    return 0.5 * float(np.sum(LAMS * m))


def oracle(noise_c):
    """Best tuned final loss with tau known, over all families."""
    best = {}
    for tau in TAUS:
        cands = []
        for e in ETA0_GRID:
            cands.append(loss_trajectory(sched_constant(tau, e), noise_c)[-1])
            cands.append(loss_trajectory(sched_cosine(tau, e, tau), noise_c)[-1])
            cands.append(loss_trajectory(sched_linear(tau, e, tau), noise_c)[-1])
            for cd in WSD_CD_GRID:
                cands.append(loss_trajectory(sched_wsd(tau, e, tau, cd), noise_c)[-1])
            for uf in U_FACTORS:
                cands.append(loss_trajectory(sched_hyperbolic(tau, e, tau, uf), noise_c)[-1])
        best[tau] = min(cands)
    return best


def contender_configs():
    cfgs = {
        "Constant": [("Constant", (e,)) for e in ETA0_GRID],
        "Cosine": [("Cosine", (e, n)) for e in ETA0_GRID for n in NPLAN_GRID],
        "Linear": [("Linear", (e, n)) for e in ETA0_GRID for n in NPLAN_GRID],
        "WSD-stop": [("WSD-stop", (e, n, cd)) for e in ETA0_GRID for n in NPLAN_GRID
                     for cd in WSD_CD_GRID],
        "HyperbolicLR": [("HyperbolicLR", (e, n, uf)) for e in ETA0_GRID
                         for n in NPLAN_GRID for uf in U_FACTORS],
        "WSD-branch": [("WSD-branch", (e,)) for e in ETA0_GRID],
    }
    return cfgs


def losses_over_taus(kind, params, noise_c):
    """Loss at each stop time tau for one committed configuration."""
    if kind == "Constant":
        traj = loss_trajectory(sched_constant(TAU_MAX, *params), noise_c)
        return {tau: traj[tau - 1] for tau in TAUS}
    if kind == "Cosine":
        traj = loss_trajectory(sched_cosine(TAU_MAX, *params), noise_c)
        return {tau: traj[tau - 1] for tau in TAUS}
    if kind == "Linear":
        traj = loss_trajectory(sched_linear(TAU_MAX, *params), noise_c)
        return {tau: traj[tau - 1] for tau in TAUS}
    if kind == "WSD-stop":
        traj = loss_trajectory(sched_wsd(TAU_MAX, *params), noise_c)
        return {tau: traj[tau - 1] for tau in TAUS}
    if kind == "HyperbolicLR":
        traj = loss_trajectory(sched_hyperbolic(TAU_MAX, *params), noise_c)
        return {tau: traj[tau - 1] for tau in TAUS}
    if kind == "WSD-branch":
        eta0 = params[0]
        stable = sched_constant(TAU_MAX, eta0)
        out = {}
        for tau in TAUS:
            m = state_at(stable, tau, noise_c)
            out[tau] = branch_cooldown_loss(m.copy(), eta0,
                                            max(1, int(BRANCH_FRAC * tau)), noise_c)
        return out
    raise ValueError(kind)


def main():
    noise_c = 0.3
    orc = oracle(noise_c)

    results = {}
    for name, cfgs in contender_configs().items():
        best_cfg, best_minimax, best_curve = None, np.inf, None
        for kind, params in cfgs:
            lt = losses_over_taus(kind, params, noise_c)
            regrets = np.array([lt[tau] / orc[tau] - 1 for tau in TAUS])
            mm = regrets.max()
            if mm < best_minimax:
                best_minimax, best_cfg, best_curve = mm, params, regrets
        results[name] = (best_cfg, best_minimax, best_curve)

    print(f"Anytime protocol at noise c={noise_c}, tau in [{TAU_MIN}, {TAU_MAX}]")
    print(f"{'Contender':<14} {'minimax regret':<16} {'mean regret':<14} committed config")
    for name, (cfg, mm, curve) in results.items():
        extra = "  (+20% compute, checkpoint infra)" if name == "WSD-branch" else ""
        print(f"{name:<14} {100*mm:<16.1f} {100*curve.mean():<14.1f} {cfg}{extra}")

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(results)))
        for (name, (_, _, curve)), color in zip(results.items(), colors):
            ls = ":" if name == "WSD-branch" else ("--" if name == "Constant" else "-")
            ax.plot(TAUS, 100 * curve, color=color, linewidth=1, linestyle=ls,
                    label=name)
        ax.set_xlabel(r"Stop time $\tau$")
        ax.set_ylabel(r"Anytime regret vs oracle [\%]")
        ax.set_yscale("log")
        ax.legend(fontsize=5)
        plt.tight_layout()
        fig.savefig("../figs/anytime_regret.png", dpi=600, bbox_inches="tight")
        plt.close()
    print("\nSaved: figs/anytime_regret.png")

    print("\nRobustness over noise:")
    for c in [0.05, 1.0]:
        orc_c = oracle(c)
        line = []
        for name, cfgs in contender_configs().items():
            best_mm = np.inf
            for kind, params in cfgs:
                lt = losses_over_taus(kind, params, c)
                mm = max(lt[tau] / orc_c[tau] - 1 for tau in TAUS)
                best_mm = min(best_mm, mm)
            line.append(f"{name} {100*best_mm:.0f}%")
        print(f"  c={c}: " + ", ".join(line))


if __name__ == "__main__":
    main()
