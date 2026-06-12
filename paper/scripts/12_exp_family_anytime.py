"""
Follow-up to script 11: the dense-grid free-form anytime optimum looks like
plateau + geometric decay. Test the two named families that embody that shape:
- Exponential: plateau then eta0 * 0.5^((t - plateau)/half_life), held implicitly by shape.
- ExpHyperbolicLR: log-LR follows the hyperbolic curve (the paper's second scheduler);
  near N ~ U it degenerates to geometric decay with an anchored endpoint.
Both tuned fairly (minimax over the dense tau grid) against the cached oracle of script 11.
"""
import numpy as np

N_MODES = 200
LAMS = 1.0 / np.arange(1, N_MODES + 1)
ETA_RATIO = 1e-3
TAU_MIN, TAU_MAX = 50, 200
TAUS = np.arange(TAU_MIN, TAU_MAX + 1)
NOISE_C = 0.3

d = np.load("../figs/dense_oracle_c0.3.npz")
ORACLE = dict(zip(d["taus"].tolist(), d["vals"].tolist()))


def loss_trajectory(etas):
    m = np.ones(N_MODES)
    out = np.empty(len(etas))
    for k, eta in enumerate(etas):
        dd = 1 - eta * LAMS
        m = dd * dd * m + eta * eta * NOISE_C * LAMS
        out[k] = 0.5 * float(np.sum(LAMS * m))
    return out


def minimax_of(etas):
    traj = loss_trajectory(etas)
    rs = np.array([traj[t - 1] / ORACLE[t] - 1 for t in TAUS])
    return float(rs.max()), float(rs.mean())


def sched_exponential(total, eta0, half_life, plateau):
    t = np.arange(total, dtype=float)
    return eta0 * 0.5 ** (np.maximum(t - plateau, 0.0) / half_life)


def sched_exphyperbolic(total, eta0, nplan, ufac):
    Uv = ufac * nplan
    t = np.arange(nplan, dtype=float)
    em = ETA_RATIO * eta0
    g = np.sqrt(np.maximum((Uv - t) ** 2 - (Uv - nplan) ** 2, 0.0)) / Uv
    g0 = np.sqrt(Uv ** 2 - (Uv - nplan) ** 2) / Uv
    e = eta0 * (eta0 / em) ** (g - g0)
    if total > nplan:
        e = np.concatenate([e, np.full(total - nplan, e[-1])])
    return e[:total]


def main():
    print("Dense anytime game, exponential-shape families (c=0.3):")

    best = (np.inf, None, None)
    for e0 in [1.0, 1.3, 1.6, 1.9]:
        for hl in np.geomspace(15, 200, 14):
            for pl in [0, 10, 20, 30, 45]:
                mm, mean = minimax_of(sched_exponential(TAU_MAX, e0, hl, pl))
                if mm < best[0]:
                    best = (mm, mean, (e0, round(float(hl), 1), pl))
    print(f"  Exponential     minimax {100*best[0]:6.1f}%  mean {100*best[1]:6.1f}%  "
          f"(eta0, half-life, plateau) = {best[2]}")

    best = (np.inf, None, None)
    for e0 in np.geomspace(0.4, 1.9, 12):
        for npl in [100, 125, 150, 175, 200, 250, 300]:
            for uf in [1.02, 1.05, 1.1, 1.2, 1.5]:
                mm, mean = minimax_of(sched_exphyperbolic(TAU_MAX, e0, npl, uf))
                if mm < best[0]:
                    best = (mm, mean, (round(float(e0), 3), npl, uf))
    print(f"  ExpHyperbolicLR minimax {100*best[0]:6.1f}%  mean {100*best[1]:6.1f}%  "
          f"(eta0, N_plan, U/N) = {best[2]}")

    print("  reference: free-form 9.3% / 8.6%; HyperbolicLR 29.9% / 17.0%; "
          "WSD-branch ~9.1% (+20% compute, sparse-grid value)")


if __name__ == "__main__":
    main()
