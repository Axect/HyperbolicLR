"""
Shape discrimination for the free-form anytime optimum: over the narrow window [50, 200]
(4x dynamic range) an exponential decay is indistinguishable from a steep power law.
Re-run the dense anytime game on the wide window [20, 640] (32x) and fit the optimum's
stop-window envelope in semilog (exponential) vs log-log (power law) coordinates.
Also computes a spot-check "true oracle" (free-form optimal control per tau) at a few taus
to quantify how much the family-based oracle overestimates the oracle loss.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import scienceplots

N_MODES = 200
LAMS_NP = 1.0 / np.arange(1, N_MODES + 1)
LAMS = torch.tensor(LAMS_NP, dtype=torch.float64)
ETA_RATIO = 1e-3
TAU_MIN, TAU_MAX = 20, 640
TAUS = np.arange(TAU_MIN, TAU_MAX + 1, 5)
NOISE_C = 0.3
ETA_CAP = 1.9

ETA0_GRID = np.geomspace(0.05, 1.9, 12)
NPLAN_GRID = [20, 40, 80, 160, 320, 640]
U_FACTORS = [1.05, 1.2, 1.5, 2.5]
WSD_CD_GRID = [0.2, 0.5]

torch.manual_seed(7)


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


def oracle_family():
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
    return best


def true_oracle_at(tau, n_iter=3000):
    """Free-form optimal control of the final loss for one known horizon tau."""
    e = np.clip(sched_linear(tau, 1.0, tau), 1e-4, ETA_CAP - 1e-4)
    z = torch.tensor(np.log(e / (ETA_CAP - e)), requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.02)
    for _ in range(n_iter):
        opt.zero_grad()
        etas = ETA_CAP * torch.sigmoid(z)
        m = torch.ones(N_MODES, dtype=torch.float64)
        for k in range(tau):
            d = 1 - etas[k] * LAMS
            m = d * d * m + etas[k] ** 2 * NOISE_C * LAMS
        (0.5 * torch.sum(LAMS * m)).backward()
        opt.step()
    with torch.no_grad():
        etas = ETA_CAP * torch.sigmoid(z)
        m = torch.ones(N_MODES, dtype=torch.float64)
        for k in range(tau):
            d = 1 - etas[k] * LAMS
            m = d * d * m + etas[k] ** 2 * NOISE_C * LAMS
        return 0.5 * float(torch.sum(LAMS * m))


def optimize_freeform(orc, n_iter=5000):
    orc_t = torch.tensor([orc[t] for t in TAUS], dtype=torch.float64)
    tau_idx = torch.tensor(TAUS - 1, dtype=torch.long)

    def regrets(etas):
        m = torch.ones(N_MODES, dtype=torch.float64)
        losses = []
        for k in range(TAU_MAX):
            d = 1 - etas[k] * LAMS
            m = d * d * m + etas[k] ** 2 * NOISE_C * LAMS
            losses.append(0.5 * torch.sum(LAMS * m))
        return torch.stack(losses)[tau_idx] / orc_t - 1

    best_val, best_etas = np.inf, None
    inits = {
        "geometric": 1.9 * 0.5 ** (np.arange(TAU_MAX) / 100.0),
        "powerlaw": 1.9 / (1 + np.arange(TAU_MAX) / 15.0),
    }
    for name, e in inits.items():
        e = np.clip(e, 1e-4, ETA_CAP - 1e-4)
        z = torch.tensor(np.log(e / (ETA_CAP - e)), requires_grad=True)
        opt = torch.optim.Adam([z], lr=0.02)
        for it in range(n_iter):
            opt.zero_grad()
            r = regrets(ETA_CAP * torch.sigmoid(z))
            beta = 50.0 + 450.0 * it / n_iter
            (torch.logsumexp(beta * r, dim=0) / beta).backward()
            opt.step()
        with torch.no_grad():
            etas = (ETA_CAP * torch.sigmoid(z)).numpy()
            val = float(regrets(torch.tensor(etas)).max())
        print(f"  init={name}: minimax {100*val:.1f}%")
        if val < best_val:
            best_val, best_etas = val, etas
    return best_val, best_etas


def main():
    print("Wide-window dense anytime game [20, 640], c=0.3")
    orc = oracle_family()
    np.savez("../figs/wide_oracle.npz", taus=np.array(list(orc.keys())),
             vals=np.array(list(orc.values())))

    print("Spot-check: family oracle vs true optimal-control oracle")
    for tau in [40, 160, 640]:
        to = true_oracle_at(tau)
        print(f"  tau={tau:>4}: family {orc[tau]:.4e}  true {to:.4e}  "
              f"family/true = {orc[tau]/to:.3f}")

    print("Free-form minimax optimization:")
    ff_val, ff_etas = optimize_freeform(orc)
    print(f"  wide-window free-form minimax: {100*ff_val:.1f}%")
    np.savez("../figs/wide_freeform.npz", etas=ff_etas)

    # shape fits on the stop window
    st = np.arange(TAU_MIN, TAU_MAX)
    y = np.log(ff_etas[TAU_MIN:TAU_MAX])
    semilog = np.polyfit(st, y, 1)
    loglog = np.polyfit(np.log(st), y, 1)
    res_semi = float(np.mean((y - np.polyval(semilog, st)) ** 2))
    res_log = float(np.mean((y - np.polyval(loglog, np.log(st))) ** 2))
    print(f"Shape fit on [{TAU_MIN},{TAU_MAX}]:")
    print(f"  exponential fit: half-life {np.log(2)/(-semilog[0]):.1f} steps, residual {res_semi:.4f}")
    print(f"  power-law fit:   exponent {loglog[0]:.2f}, residual {res_log:.4f}")
    print(f"  verdict: {'POWER LAW' if res_log < res_semi else 'EXPONENTIAL'} fits better "
          f"(residual ratio {res_semi/res_log:.2f}x)")

    with plt.style.context(["science", "nature"]):
        fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5))
        axes[0].plot(np.arange(TAU_MAX), ff_etas, color="tab:red", linewidth=0.8)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Step $t$")
        axes[0].set_ylabel(r"$\eta_t$")
        axes[0].set_title("Wide-window optimum (semilog)", fontsize=7)
        axes[1].plot(np.arange(1, TAU_MAX + 1), ff_etas, color="tab:red", linewidth=0.8)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Step $t$")
        axes[1].set_title("Wide-window optimum (log-log)", fontsize=7)
        plt.tight_layout()
        fig.savefig("../figs/wide_window_shape.png", dpi=600, bbox_inches="tight")
        plt.close()
    print("Saved: figs/wide_window_shape.png")


if __name__ == "__main__":
    main()
