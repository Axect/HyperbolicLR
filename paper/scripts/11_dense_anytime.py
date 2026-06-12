"""
Anytime game v2 with a DENSE stop grid (every step in [50, 200]) to rule out the
possibility that the sawtooth optimum of script 10 merely exploits the sparse tau grid.

- Oracle(tau) for every tau in [50, 200] (cached to npz).
- Re-evaluate the committed-family contenders of script 09 under dense minimax.
- Re-optimize the free-form schedule against the dense grid.
- Add a named "Sawtooth" family: exponential envelope with periodic geometric dips
  (a single-run emulation of WSD-branch / CyclicLR exp_range), fairly tuned like others.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import scienceplots

N_MODES = 200
LAMS_NP = 1.0 / np.arange(1, N_MODES + 1)
ETA_RATIO = 1e-3
TAU_MIN, TAU_MAX = 50, 200
TAUS = np.arange(TAU_MIN, TAU_MAX + 1)          # dense: every step
NOISE_C = 0.3
ETA_CAP = 1.9

ETA0_GRID = np.geomspace(0.05, 1.9, 12)
NPLAN_GRID = [50, 75, 100, 125, 150, 175, 200]
U_FACTORS = [1.05, 1.2, 1.5, 2.5]
WSD_CD_GRID = [0.2, 0.5]

torch.manual_seed(7)
LAMS = torch.tensor(LAMS_NP, dtype=torch.float64)
ORACLE_CACHE = "../figs/dense_oracle_c0.3.npz"


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


def sched_sawtooth(total, eta0, half_life, period, depth, plateau):
    t = np.arange(total, dtype=float)
    env = eta0 * 0.5 ** (np.maximum(t - plateau, 0.0) / half_life)
    phase = ((t - plateau) % period) / period
    phase[t < plateau] = 0.0
    dip = depth ** phase
    return env * dip


def oracle_dense():
    if os.path.exists(ORACLE_CACHE):
        d = np.load(ORACLE_CACHE)
        return dict(zip(d["taus"].tolist(), d["vals"].tolist()))
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
        best[int(tau)] = min(cands)
    np.savez(ORACLE_CACHE, taus=np.array(list(best.keys())),
             vals=np.array(list(best.values())))
    return best


def minimax_of(etas, orc):
    traj = loss_trajectory_np(etas)
    rs = np.array([traj[t - 1] / orc[t] - 1 for t in TAUS])
    return float(rs.max()), float(rs.mean()), rs


def tune_family(builder, grids, orc):
    best = (np.inf, None, None, None)
    for params in grids:
        mm, mean, rs = minimax_of(builder(TAU_MAX, *params), orc)
        if mm < best[0]:
            best = (mm, mean, params, rs)
    return best


def optimize_freeform(orc, n_iter=4000):
    orc_t = torch.tensor([orc[t] for t in TAUS], dtype=torch.float64)
    tau_idx = torch.tensor(TAUS - 1, dtype=torch.long)

    def fwd(etas):
        m = torch.ones(N_MODES, dtype=torch.float64)
        losses = []
        for k in range(TAU_MAX):
            d = 1 - etas[k] * LAMS
            m = d * d * m + etas[k] ** 2 * NOISE_C * LAMS
            losses.append(0.5 * torch.sum(LAMS * m))
        return torch.stack(losses)[tau_idx] / orc_t - 1

    inits = {
        "hyperbolic": sched_hyperbolic(TAU_MAX, 1.365, 100, 1.2),
        "sawtooth": sched_sawtooth(TAU_MAX, 1.9, 60.0, 10, 0.35, 20),
        "linear": sched_linear(TAU_MAX, 0.98, 175),
    }
    best_val, best_etas = np.inf, None
    for name, e in inits.items():
        e = np.clip(e, 1e-4, ETA_CAP - 1e-4)
        z = torch.tensor(np.log(e / (ETA_CAP - e)), requires_grad=True)
        opt = torch.optim.Adam([z], lr=0.02)
        for it in range(n_iter):
            opt.zero_grad()
            r = fwd(ETA_CAP * torch.sigmoid(z))
            beta = 50.0 + 450.0 * it / n_iter
            (torch.logsumexp(beta * r, dim=0) / beta).backward()
            opt.step()
        with torch.no_grad():
            etas = (ETA_CAP * torch.sigmoid(z)).numpy()
        val = minimax_of(etas, orc)[0]
        print(f"  freeform init={name}: minimax {100*val:.1f}%")
        if val < best_val:
            best_val, best_etas = val, etas
    return best_val, best_etas


def main():
    orc = oracle_dense()
    print("Dense-grid anytime game (tau = every step in [50,200]), c=0.3")

    fams = {
        "Constant": (lambda total, e: np.full(total, e),
                     [(e,) for e in ETA0_GRID]),
        "Cosine": (sched_cosine, [(e, n) for e in ETA0_GRID for n in NPLAN_GRID]),
        "Linear": (sched_linear, [(e, n) for e in ETA0_GRID for n in NPLAN_GRID]),
        "WSD-stop": (sched_wsd, [(e, n, cd) for e in ETA0_GRID for n in NPLAN_GRID
                                 for cd in WSD_CD_GRID]),
        "HyperbolicLR": (sched_hyperbolic, [(e, n, uf) for e in ETA0_GRID
                                            for n in NPLAN_GRID for uf in U_FACTORS]),
        "Sawtooth": (sched_sawtooth,
                     [(e, hl, p, d, pl) for e in [1.2, 1.6, 1.9]
                      for hl in [40.0, 60.0, 90.0, 130.0]
                      for p in [5, 10, 20]
                      for d in [0.2, 0.35, 0.5]
                      for pl in [0, 20, 40]]),
    }
    results = {}
    for name, (builder, grids) in fams.items():
        mm, mean, params, rs = tune_family(builder, grids, orc)
        results[name] = (mm, mean, params, rs)
        print(f"  {name:<13} minimax {100*mm:6.1f}%  mean {100*mean:6.1f}%  cfg {params}")

    print("Optimizing free-form on dense grid:")
    ff_val, ff_etas = optimize_freeform(orc)
    ff_mm, ff_mean, ff_rs = minimax_of(ff_etas, orc)
    print(f"  Free-form     minimax {100*ff_mm:6.1f}%  mean {100*ff_mean:6.1f}%")

    with plt.style.context(["science", "nature"]):
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5))
        st = np.arange(TAU_MAX)
        ax = axes[0]
        ax.plot(st, ff_etas, color="tab:red", linewidth=0.8, label="Free-form optimum")
        ax.plot(st, sched_sawtooth(TAU_MAX, *results["Sawtooth"][2]), color="tab:blue",
                linewidth=0.8, linestyle="--", label="Tuned Sawtooth")
        ax.plot(st, sched_hyperbolic(TAU_MAX, *results["HyperbolicLR"][2]),
                color="tab:green", linewidth=0.8, linestyle="-.", label="Tuned HyperbolicLR")
        ax.set_yscale("log")
        ax.set_xlabel("Step $t$")
        ax.set_ylabel(r"$\eta_t$")
        ax.legend(fontsize=5)
        ax.set_title("Schedules (dense game)", fontsize=7)
        ax = axes[1]
        for name in ["Cosine", "WSD-stop", "HyperbolicLR", "Sawtooth"]:
            ax.plot(TAUS, 100 * results[name][3], linewidth=0.8, label=name)
        ax.plot(TAUS, 100 * ff_rs, linewidth=0.8, color="tab:red", label="Free-form")
        ax.set_yscale("log")
        ax.set_xlabel(r"Stop time $\tau$")
        ax.set_ylabel(r"Regret [\%]")
        ax.legend(fontsize=5)
        ax.set_title("Dense anytime regret", fontsize=7)
        plt.tight_layout()
        fig.savefig("../figs/dense_anytime.png", dpi=600, bbox_inches="tight")
        plt.close()
    print("Saved: figs/dense_anytime.png")
    np.savez("../figs/dense_anytime_freeform.npz", etas=ff_etas)


if __name__ == "__main__":
    main()
