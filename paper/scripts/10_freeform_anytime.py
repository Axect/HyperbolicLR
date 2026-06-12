"""
Free-form anytime-optimal schedule: optimize the 200-step LR sequence itself (no family
assumption) to minimize anytime regret on the exact quadratic dynamics, and compare the
resulting shape and value against the best committed HyperbolicLR from script 09.

- Decision variable: eta_k = ETA_CAP * sigmoid(z_k), k = 0..TAU_MAX-1 (positivity and
  stability bound built in; monotonicity NOT enforced on purpose).
- Objective A (minimax): softmax-smoothed max over tau of regret, temperature annealed,
  final evaluation with the true max.
- Objective B (mean): average regret over tau.
- Oracle(tau) and the tau grid are identical to scripts/09 so regret values are directly
  comparable.
- Byproduct: best hyperbolic-family fit to the free-form optimum (grid search), giving the
  "how close to optimal is the closed form" gap.
Runs on CPU deliberately (GPU is occupied by another project).
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import scienceplots

N_MODES = 200
LAMS_NP = 1.0 / np.arange(1, N_MODES + 1)
ETA_RATIO = 1e-3
TAU_MIN, TAU_MAX = 50, 200
TAUS = np.arange(TAU_MIN, TAU_MAX + 1, 10)
NOISE_C = 0.3
ETA_CAP = 1.9

ETA0_GRID = np.geomspace(0.05, 1.9, 12)
NPLAN_GRID = [50, 75, 100, 125, 150, 175, 200]
U_FACTORS = [1.05, 1.2, 1.5, 2.5]
WSD_CD_GRID = [0.2, 0.5]

torch.manual_seed(7)
DEV = "cpu"
LAMS = torch.tensor(LAMS_NP, dtype=torch.float64, device=DEV)


# ── exact dynamics (numpy, for oracle/baselines; identical to script 09) ─────

def loss_trajectory_np(etas):
    m = np.ones(N_MODES)
    out = np.empty(len(etas))
    for k, eta in enumerate(etas):
        d = 1 - eta * LAMS_NP
        m = d * d * m + eta * eta * NOISE_C * LAMS_NP
        out[k] = 0.5 * float(np.sum(LAMS_NP * m))
    return out


def hold_tail(e, total):
    if len(e) >= total:
        return e[:total]
    return np.concatenate([e, np.full(total - len(e), e[-1])])


def sched_cosine(total, eta0, nplan):
    t = np.arange(nplan)
    em = ETA_RATIO * eta0
    return hold_tail(em + (eta0 - em) * (1 + np.cos(np.pi * t / nplan)) / 2, total)


def sched_linear(total, eta0, nplan):
    t = np.arange(nplan)
    em = ETA_RATIO * eta0
    return hold_tail(eta0 + (em - eta0) * t / nplan, total)


def sched_wsd(total, eta0, nplan, cd):
    t = np.arange(nplan, dtype=float)
    em = ETA_RATIO * eta0
    start = (1 - cd) * nplan
    e = np.full(nplan, eta0)
    mask = t > start
    e[mask] = em + (eta0 - em) * (1 - np.sqrt((t[mask] - start) / (cd * nplan)))
    return hold_tail(e, total)


def sched_hyperbolic(total, eta0, nplan, ufac):
    Uv = ufac * nplan
    t = np.arange(nplan, dtype=float)
    em = ETA_RATIO * eta0
    g = np.sqrt(np.maximum((Uv - t) ** 2 - (Uv - nplan) ** 2, 0.0)) / Uv
    g0 = np.sqrt(Uv ** 2 - (Uv - nplan) ** 2) / Uv
    return hold_tail(eta0 + (eta0 - em) * (g - g0), total)


def oracle():
    best = {}
    for tau in TAUS:
        cands = []
        for e in ETA0_GRID:
            cands.append(loss_trajectory_np(np.full(tau, e))[-1])
            cands.append(loss_trajectory_np(sched_cosine(tau, e, tau))[-1])
            cands.append(loss_trajectory_np(sched_linear(tau, e, tau))[-1])
            for cd in WSD_CD_GRID:
                cands.append(loss_trajectory_np(sched_wsd(tau, e, tau, cd))[-1])
            for uf in U_FACTORS:
                cands.append(loss_trajectory_np(sched_hyperbolic(tau, e, tau, uf))[-1])
        best[tau] = min(cands)
    return best


# ── differentiable dynamics and optimization ─────────────────────────────────

def losses_at_taus_torch(etas):
    m = torch.ones(N_MODES, dtype=torch.float64, device=DEV)
    out = []
    tau_set = set(TAUS.tolist())
    for k in range(TAU_MAX):
        d = 1 - etas[k] * LAMS
        m = d * d * m + etas[k] ** 2 * NOISE_C * LAMS
        if (k + 1) in tau_set:
            out.append(0.5 * torch.sum(LAMS * m))
    return torch.stack(out)


def regrets_torch(etas, orc_t):
    return losses_at_taus_torch(etas) / orc_t - 1


def init_z(kind):
    if kind == "hyperbolic":
        e = sched_hyperbolic(TAU_MAX, 1.365, 100, 1.2)
    elif kind == "linear":
        e = sched_linear(TAU_MAX, 0.98, 175)
    elif kind == "constant":
        e = np.full(TAU_MAX, 0.36)
    else:
        rng = np.random.default_rng(0)
        e = np.exp(rng.uniform(np.log(0.05), np.log(1.5), TAU_MAX))
    e = np.clip(e, 1e-4, ETA_CAP - 1e-4)
    return torch.tensor(np.log(e / (ETA_CAP - e)), dtype=torch.float64,
                        device=DEV, requires_grad=True)


def optimize(objective, orc_t, n_iter=4000):
    best_val, best_etas = np.inf, None
    for kind in ["hyperbolic", "linear", "constant", "random"]:
        z = init_z(kind)
        opt = torch.optim.Adam([z], lr=0.02)
        for it in range(n_iter):
            opt.zero_grad()
            etas = ETA_CAP * torch.sigmoid(z)
            r = regrets_torch(etas, orc_t)
            if objective == "minimax":
                beta = 50.0 + 450.0 * it / n_iter
                obj = torch.logsumexp(beta * r, dim=0) / beta
            else:
                obj = r.mean()
            obj.backward()
            opt.step()
        with torch.no_grad():
            etas = ETA_CAP * torch.sigmoid(z)
            r = regrets_torch(etas, orc_t)
            val = float(r.max()) if objective == "minimax" else float(r.mean())
        if val < best_val:
            best_val = val
            best_etas = etas.detach().cpu().numpy()
    return best_val, best_etas


def fit_hyperbolic(target_etas, orc):
    """Closest committed hyperbolic config to the free-form optimum, by minimax regret."""
    best = (np.inf, None)
    for e0 in np.geomspace(0.2, 1.9, 24):
        for npl in range(60, 201, 10):
            for uf in [1.02, 1.05, 1.1, 1.2, 1.35, 1.5, 2.0, 2.5]:
                e = sched_hyperbolic(TAU_MAX, e0, npl, uf)
                traj = loss_trajectory_np(e)
                mm = max(traj[t - 1] / orc[t] - 1 for t in TAUS)
                if mm < best[0]:
                    best = (mm, (e0, npl, uf))
    return best


def main():
    orc = oracle()
    orc_t = torch.tensor([orc[t] for t in TAUS], dtype=torch.float64, device=DEV)

    mm_val, mm_etas = optimize("minimax", orc_t)
    mean_val, mean_etas = optimize("mean", orc_t)

    with torch.no_grad():
        mm_mean = float(regrets_torch(torch.tensor(mm_etas), orc_t).mean())
        mean_mm = float(regrets_torch(torch.tensor(mean_etas), orc_t).max())

    print(f"Free-form anytime-optimal at c={NOISE_C} (tau in [{TAU_MIN},{TAU_MAX}]):")
    print(f"  minimax-optimal: minimax {100*mm_val:.1f}%  (its mean {100*mm_mean:.1f}%)")
    print(f"  mean-optimal:    mean {100*mean_val:.1f}%  (its minimax {100*mean_mm:.1f}%)")
    print(f"  reference (script 09): HyperbolicLR committed: minimax 29.9%, mean 17.7%;"
          f" WSD-branch 9.1% (+20% compute)")

    fit_mm, fit_cfg = fit_hyperbolic(mm_etas, orc)
    print(f"  best hyperbolic-family config (fine grid): minimax {100*fit_mm:.1f}%"
          f"  at (eta0={fit_cfg[0]:.3f}, N_plan={fit_cfg[1]}, U={fit_cfg[2]:.2f}N)")

    hyp_ref = sched_hyperbolic(TAU_MAX, *([fit_cfg[0], fit_cfg[1], fit_cfg[2]]))

    with plt.style.context(["science", "nature"]):
        fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5), sharey=True)
        steps = np.arange(TAU_MAX)
        for ax, etas, ttl in [(axes[0], mm_etas, "Minimax-optimal"),
                              (axes[1], mean_etas, "Mean-optimal")]:
            ax.plot(steps, etas, color="tab:red", linewidth=1, label="Free-form optimum")
            ax.plot(steps, hyp_ref, color="tab:green", linewidth=1, linestyle="--",
                    label="Best hyperbolic fit")
            for tau in TAUS:
                ax.axvline(tau, color="gray", linewidth=0.3, alpha=0.3)
            ax.set_yscale("log")
            ax.set_xlabel("Step $t$")
            ax.set_title(ttl, fontsize=7)
        axes[0].set_ylabel(r"$\eta_t$")
        axes[0].legend(fontsize=5)
        plt.tight_layout()
        fig.savefig("../figs/freeform_anytime_optimum.png", dpi=600,
                    bbox_inches="tight")
        plt.close()
    print("Saved: figs/freeform_anytime_optimum.png")

    np.savez("../figs/freeform_anytime_optimum.npz", minimax=mm_etas, mean=mean_etas)


if __name__ == "__main__":
    main()
