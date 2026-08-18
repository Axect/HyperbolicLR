"""
Does the exactly solvable quadratic model reproduce budget-extrapolation failure? No.

Setting. Diagonal quadratic with curvature spectrum {lambda_i} and gradient noise sigma^2.
Exact second-moment recursion per mode:  m_i(t+1) = (1 - eta_t lambda_i)^2 m_i(t) + eta_t^2 sigma^2,
loss L(t) = sum_i lambda_i m_i(t) / 2. A mode expands iff |1 - eta lambda_i| > 1, i.e.
eta > 2 / lambda_i, so the stiffest mode sets a stability threshold theta* = 2 / lambda_max.

Hypothesis under test. Tuning at a small budget selects eta_0 above theta* (the stiff mode's
transient blow-up is short lived and is outweighed by faster progress on the rest of the
spectrum). Extending the budget multiplies the dwell time T_{theta*}(N) above the threshold for
every rescaling family, so the blow-up should compound and the tuned configuration should
degrade.

Measured outcome (2026-08-17). The first half is confirmed and the second half is false. Tuning
at 50 steps does select eta_0 = 2.00 = 2 theta* for E, P, L and EH, and the stiff mode does
explode: for E its second moment reaches 6.7e8 by step 20. But the blow-up is entirely
reversible. Once the schedule anneals through eta = 1 / lambda_max the contraction factor
(1 - eta lambda_max)^2 passes through exactly 0, so the damaged mode is annihilated in a single
step, and every family ends up better at a longer budget. No family fails.

Consequence for the paper. Budget-extrapolation failure is not a property of the schedule alone:
it needs irreversibility, which a linear-Gaussian (or any convex quadratic) surrogate does not
have. This is a limitation of the convex-analysis line of work on schedules, and it is why the
claim has to be established on real nonconvex training (operator learning) rather than in the
solvable model. The solvable model still supplies the closed-form dwell-time scaling, which is
the part that does transfer.

Usage: python 18_dwell_extrapolation.py [--n-modes K] [--sigma S]
"""
import argparse

import numpy as np


def spectrum(k, lam_min=0.01, lam_max=2.0):
    return np.logspace(np.log10(lam_min), np.log10(lam_max), k)


def lr_curve(name, params, n):
    """Learning rate sequence of one family over a budget of n steps."""
    eta0 = params["eta0"]
    t = np.arange(n)
    if name == "N":
        return np.full(n, eta0)
    if name == "E":
        return eta0 * params["gamma"] ** t
    if name == "P":
        return eta0 * (1.0 - t / n) ** params["power"]
    if name == "C":
        emin = params["eta_min"]
        return emin + 0.5 * (eta0 - emin) * (1.0 + np.cos(np.pi * t / n))
    if name == "L":
        emin = params["eta_min"]
        return eta0 + (emin - eta0) * t / n
    if name == "WSD":
        emin = params["eta_min"]
        warm = max(1, n // 10)
        cool = max(1, int(round(params["cooldown_frac"] * n)))
        out = np.full(n, eta0, dtype=float)
        out[:warm] = eta0 * (np.arange(1, warm + 1) / warm)
        s = np.arange(1, cool + 1) / cool
        out[n - cool:] = eta0 + (emin - eta0) * np.sqrt(s)
        return out
    if name == "H":
        # HyperbolicLR: eta_N(t) = eta0 + delta * (g(t) - g(0)),
        # g(t) = sqrt((N - t)(2U - N - t)) / U, delta = eta0 - eta_inf.
        u, einf = params["U"], params["eta_min"]
        u = max(u, n)
        g = np.sqrt(np.maximum((n - t) * (2 * u - n - t), 0.0)) / u
        return eta0 + (eta0 - einf) * (g - g[0])
    if name == "EH":
        # ExpHyperbolicLR: the same curve in log space.
        u, einf = params["U"], params["eta_min"]
        u = max(u, n)
        g = np.sqrt(np.maximum((n - t) * (2 * u - n - t), 0.0)) / u
        return eta0 * (eta0 / einf) ** (g - g[0])
    raise ValueError(name)


def run(lrs, lam, sigma2, m0=1.0):
    """Exact second-moment propagation. Returns the loss trajectory."""
    m = np.full_like(lam, m0)
    out = np.empty(len(lrs))
    for k, eta in enumerate(lrs):
        m = (1.0 - eta * lam) ** 2 * m + eta * eta * sigma2
        out[k] = 0.5 * float(np.sum(lam * m))
    return out


GRIDS = {
    "N": {"eta0": np.logspace(-2, 0.3, 24)},
    "E": {"eta0": np.logspace(-2, 0.3, 24), "gamma": [0.90, 0.94, 0.96, 0.98, 0.99]},
    "P": {"eta0": np.logspace(-2, 0.3, 24), "power": [0.5, 1.0, 1.5, 2.0, 3.0]},
    "C": {"eta0": np.logspace(-2, 0.3, 24), "eta_min": [1e-6, 1e-4, 1e-2]},
    "L": {"eta0": np.logspace(-2, 0.3, 24), "eta_min": [1e-6, 1e-4, 1e-2]},
    "WSD": {"eta0": np.logspace(-2, 0.3, 24), "eta_min": [1e-6, 1e-4],
            "cooldown_frac": [0.1, 0.2, 0.4]},
    "H": {"eta0": np.logspace(-2, 0.3, 24), "eta_min": [1e-6, 1e-4],
          "U": [200, 250, 300, 400]},
    "EH": {"eta0": np.logspace(-2, 0.3, 24), "eta_min": [1e-6, 1e-4],
           "U": [200, 250, 300, 400]},
}


def configs(name):
    grid = GRIDS[name]
    keys = list(grid)
    out = [{}]
    for k in keys:
        out = [dict(c, **{k: v}) for c in out for v in grid[k]]
    return out


def tune(name, lam, sigma2, n_tune):
    best, best_loss = None, np.inf
    for c in configs(name):
        traj = run(lr_curve(name, c, n_tune), lam, sigma2)
        v = traj[-1]
        if np.isfinite(v) and v < best_loss:
            best, best_loss = c, v
    return best, best_loss


def dwell(name, c, n, theta):
    return int((lr_curve(name, c, n) > theta).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-modes", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--tune-budget", type=int, default=50)
    ap.add_argument("--deploy", type=int, nargs="+", default=[50, 100, 200, 400])
    args = ap.parse_args()

    lam = spectrum(args.n_modes)
    sigma2 = args.sigma ** 2
    theta = 2.0 / lam.max()
    print(f"spectrum: {args.n_modes} modes, lambda in [{lam.min():.3g}, {lam.max():.3g}], "
          f"sigma^2 = {sigma2:.3g}")
    print(f"stability threshold theta* = 2 / lambda_max = {theta:.4g}")
    print(f"tuning budget = {args.tune_budget}, deployment budgets = {args.deploy}\n")

    names = ["N", "E", "P", "C", "L", "WSD", "H", "EH"]
    head = (f"{'sched':<5}{'eta0':>9}{'eta0/theta*':>12}"
            + "".join(f"{'L(' + str(n) + ')':>12}" for n in args.deploy)
            + f"{'worst/L(tune)':>14}"
            + "".join(f"{'T*(' + str(n) + ')':>10}" for n in args.deploy))
    print("tuned at the small budget, then deployed unchanged at longer budgets")
    print(head)
    rows = []
    for name in names:
        c, l_tune = tune(name, lam, sigma2, args.tune_budget)
        losses = [run(lr_curve(name, c, n), lam, sigma2)[-1] for n in args.deploy]
        dwells = [dwell(name, c, n, theta) for n in args.deploy]
        worst = max(losses) / l_tune
        rows.append((name, c["eta0"], c["eta0"] / theta, losses, worst, dwells))
        print(f"{name:<5}{c['eta0']:>9.3g}{c['eta0'] / theta:>12.2f}"
              + "".join(f"{v:>12.3e}" for v in losses)
              + f"{worst:>14.3g}"
              + "".join(f"{d:>10d}" for d in dwells))

    print("\nT*(N) = epochs with lr above theta*, i.e. steps the stiffest mode is expanding")
    print("worst/L(tune) > 1 means extending the budget made the tuned configuration worse")

    print("\ndwell growth vs FINAL loss: no family fails, the model is reversible")
    print(f"{'sched':<5}{'T*(400)/T*(50)':>16}{'L(400)/L(50)':>14}{'verdict':>10}")
    for name, _, _, losses, _, dwells in rows:
        gr = dwells[-1] / dwells[0] if dwells[0] else float("nan")
        dm = losses[-1] / losses[0]
        print(f"{name:<5}{gr:>16.2f}{dm:>14.3g}{'FAIL' if dm > 1.05 else 'ok':>10}")

    # Where the mechanism does show up: the size of the transient excursion. In a real
    # nonconvex network this excursion is what is not recovered from.
    print("\nwhere the mechanism is visible: transient excursion peak/final")
    print(f"{'sched':<5}" + "".join(f"{'log10 exc(' + str(n) + ')':>18}" for n in args.deploy))
    xs, ys = [], []
    for name in names:
        c, _ = tune(name, lam, sigma2, args.tune_budget)
        cells = []
        for n in args.deploy:
            traj = run(lr_curve(name, c, n), lam, sigma2)
            e = float(np.log10(traj.max() / traj[-1]))
            cells.append(e)
            xs.append(dwell(name, c, n, theta))
            ys.append(e)
        print(f"{name:<5}" + "".join(f"{v:>18.2f}" for v in cells))
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    slope, icept = np.polyfit(x, y, 1)
    pred = slope * x + icept
    r2 = 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r = float(np.corrcoef(x, y)[0, 1])
    print(f"\nlaw over {len(x)} (family, budget) cells: "
          f"log10(peak/final) = {slope:.4f} * T*(N) + {icept:.3f}, R^2 = {r2:.4f}, r = {r:+.4f}")
    print("so the excursion is exponential in the dwell time, and the dwell time is what the")
    print("budget multiplies. The final loss hides it only because a quadratic recovers exactly:")
    print("the contraction factor (1 - eta lambda_max)^2 passes through 0 as the schedule anneals.")


if __name__ == "__main__":
    main()
