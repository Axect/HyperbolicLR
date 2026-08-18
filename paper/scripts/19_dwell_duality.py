"""
Threshold-domain dual of the budget-sensitivity conservation law.

Time domain (already in THEORY_REFRAMING.md): a scheduler family {eta_N} has budget
sensitivity mass
    M(N) = int_0^N d/dN eta_N(t) dt = B'(N) - eta_N(N),      B(N) = int_0^N eta_N(t) dt,
so the total amount by which the learning rate curve moves when the budget grows is fixed by
the cumulative budget and the endpoint; only its PLACEMENT IN TIME is a design choice.

Threshold domain (this script). Let the dwell time above a level theta be
    T_theta(N) = |{t in [0, N] : eta_N(t) > theta}| = eta_N^{-1}(theta)   (eta_N decreasing).
Layer-cake gives, with no assumptions,
    int_{eta_inf}^{eta_0} T_theta(N) d theta = B(N) - N eta_inf,                        (D1)
and differentiating in N,
    int_{eta_inf}^{eta_0} d/dN T_theta(N) d theta = B'(N) - eta_inf = M(N) + A(N),      (D2)
where A(N) = eta_N(N) - eta_inf is the anchoring error. So the dwell sensitivity integrated over
levels is the sensitivity mass plus the anchoring error: the same conservation law read in the
dual variable. Placement in time becomes PLACEMENT IN THE LEVEL SPECTRUM.

Why that matters. Damage is not caused by dwell at an arbitrary level. It is caused by dwell above
the instability threshold, which is comparable to the peak learning rate, not to the floor. So the
dangerous part of the spectrum is the high-theta end, and a family is budget-extrapolatable
exactly when it keeps its dwell sensitivity away from that end. This turns an abstract accounting
identity into a design criterion.

Where the published competitors land. Three baselines are carried here alongside our families,
because the criterion above is exactly what they were designed against:
  POW  Power scheduler, Shen et al., arXiv:2408.13359, Eqs. (7) and (8). Base phase
       min(eta_max, beta a t^b) with b = -0.51, absolute warmup, decay over the last 10%.
  WSQD Ma and Chen, arXiv:2607.10959, Eq. (3). Shifted inverse-square-root base
       c0 / sqrt(t + T0), then a linear cooldown over the last alpha = 0.2 of the budget.
  D2Z  linear decay to zero, the fully rescaling control (LinearLR with end_factor 0).
Both POW and WSQD have a budget-INDEPENDENT base phase, so their dwell sensitivity should sit
at the low-theta end with EH rather than at the high-theta end with C/P/L/WSD, while D2Z, whose
every level is rescaled by the budget, should sit with the rescaling families. That is the
prediction this table checks: our criterion has to classify the competitors correctly before it
can be used to claim anything about them.

Usage: python 19_dwell_duality.py [--N 100] [--eta0 4.6e-3] [--einf 2.66e-7] [--U 250]
                                  [--pow-b -0.51] [--pow-decay-frac 0.1]
                                  [--wsqd-shift 10] [--wsqd-decay-frac 0.2] [--warmup-abs 5]
"""
import argparse

import numpy as np
from scipy.optimize import brentq


def eta_of_t(name, t, n, eta0, einf, u=250.0, power=1.2443, gamma=0.9598,
             cooldown_frac=0.2, warmup_frac=0.1, pow_b=-0.51, pow_decay_frac=0.1,
             wsqd_shift=10.0, wsqd_decay_frac=0.2, warmup_abs=5.0):
    """Continuous-time learning rate curve at fraction-free time t in [0, n]."""
    t = np.asarray(t, dtype=float)
    if name == "N":
        return np.full_like(t, eta0)
    if name == "E":
        return eta0 * gamma ** t
    if name == "P":
        return einf + (eta0 - einf) * (1.0 - t / n) ** power
    if name == "C":
        return einf + 0.5 * (eta0 - einf) * (1.0 + np.cos(np.pi * t / n))
    if name == "L":
        return eta0 + (einf - eta0) * (t / n)
    if name == "WSD":
        warm = warmup_frac * n
        cool = cooldown_frac * n
        out = np.empty_like(t)
        rise = t < warm
        out[rise] = eta0 * np.maximum(t[rise], 1e-12) / warm
        mid = (t >= warm) & (t <= n - cool)
        out[mid] = eta0
        tail = t > n - cool
        s = (t[tail] - (n - cool)) / cool
        out[tail] = eta0 + (einf - eta0) * np.sqrt(s)
        return out
    if name == "D2Z":
        # Linear decay to zero. Same slope family as L, but anchored at 0 instead of einf,
        # so every level in the spectrum is rescaled by N with no exception.
        return eta0 * (1.0 - t / n)
    if name == "POW":
        # arXiv:2408.13359 Eq. (8). warm and the base curve do not scale with n (the paper
        # fixes warmup at 1B tokens while T sweeps 2B to 256B); only the decay onset does.
        # Eq. (7) is min(eta_max, beta a t^b), which with eta_max = eta0 and
        # beta a = eta0 warm^(-b) is eta0 (max(t, warm) / warm)^b.
        warm = warmup_abs
        cool = pow_decay_frac * n
        out = np.empty_like(t)
        rise = t < warm
        out[rise] = eta0 * np.maximum(t[rise], 1e-12) / warm
        base = eta0 * (np.maximum(t, warm) / warm) ** pow_b
        mid = (t >= warm) & (t <= n - cool)
        out[mid] = base[mid]
        tail = t > n - cool
        # Eq. (8) third branch. f is inherited from WSD, so use the same 1 - sqrt shape the
        # WSD branch above uses, anchored at 0 as in "annealing the learning rate to 0".
        edge = eta0 * (max(n - cool, warm) / warm) ** pow_b
        out[tail] = edge * (1.0 - np.sqrt((t[tail] - (n - cool)) / cool))
        return out
    if name == "WSQD":
        # arXiv:2607.10959 Eq. (3), with c0 = eta0 sqrt(warm + T0) so the base peaks at eta0
        # at the end of the paper's fixed 300-step (here: absolute) warmup.
        warm = warmup_abs
        cool = wsqd_decay_frac * n
        out = np.empty_like(t)
        rise = t < warm
        out[rise] = eta0 * np.maximum(t[rise], 1e-12) / warm
        base = eta0 * np.sqrt((warm + wsqd_shift) / (np.maximum(t, warm) + wsqd_shift))
        mid = (t >= warm) & (t <= n - cool)
        out[mid] = base[mid]
        tail = t > n - cool
        edge = eta0 * np.sqrt((warm + wsqd_shift)
                              / (max(n - cool, warm) + wsqd_shift))
        out[tail] = edge * (n - t[tail]) / cool
        return out
    if name in ("H", "EH"):
        uu = max(u, n)
        g = np.sqrt(np.maximum((n - t) * (2 * uu - n - t), 0.0)) / uu
        g0 = np.sqrt(n * (2 * uu - n)) / uu
        if name == "H":
            return eta0 + (eta0 - einf) * (g - g0)
        return eta0 * (eta0 / einf) ** (g - g0)
    raise ValueError(name)


def dwell(name, theta, n, eta0, einf, grid=200001, **kw):
    """T_theta(N) = measure{t in [0, N] : eta_N(t) > theta}.

    Computed as a measure rather than as an inverse function, because WSD is not monotone
    (it warms up first), so its super-level set is not an interval starting at 0.
    """
    ts = np.linspace(0.0, n, grid)
    above = eta_of_t(name, ts, n, eta0, einf, **kw) > theta
    return float(above.sum()) * (n / (grid - 1))


def budget(name, n, eta0, einf, **kw):
    ts = np.linspace(0.0, n, 20001)
    return float(np.trapezoid(eta_of_t(name, ts, n, eta0, einf, **kw), ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=float, default=100.0)
    ap.add_argument("--dN", type=float, default=1.0)
    ap.add_argument("--eta0", type=float, default=4.6e-3)
    ap.add_argument("--einf", type=float, default=2.66e-7)
    ap.add_argument("--U", type=float, default=250.0)
    # Competitor shape parameters, at the values their own papers recommend:
    # b = -0.51 and a 10% decay phase for POW (arXiv:2408.13359 Section 4 and Section 3),
    # alpha = 0.2 for WSQD (arXiv:2607.10959 Section 3.1), and T0 = the pilot horizon
    # (Section 3.4), which for this project's HPO is 10 epochs. warmup-abs is absolute
    # rather than a fraction of N because both papers hold the warmup fixed across horizons;
    # it matches scheduler_config.ABS_WARMUP_EPOCHS, which is 5 so that the ramp still fits
    # inside the smallest budget this project tunes at.
    ap.add_argument("--pow-b", type=float, default=-0.51)
    ap.add_argument("--pow-decay-frac", type=float, default=0.1)
    ap.add_argument("--wsqd-shift", type=float, default=10.0)
    ap.add_argument("--wsqd-decay-frac", type=float, default=0.2)
    ap.add_argument("--warmup-abs", type=float, default=5.0)
    args = ap.parse_args()

    n, dn, eta0, einf = args.N, args.dN, args.eta0, args.einf
    # D2Z sits next to L (both rescale every level), POW and WSQD next to their parent WSD.
    names = ["N", "E", "P", "C", "L", "D2Z", "WSD", "POW", "WSQD", "H", "EH"]
    kw = {"u": args.U, "pow_b": args.pow_b, "pow_decay_frac": args.pow_decay_frac,
          "wsqd_shift": args.wsqd_shift, "wsqd_decay_frac": args.wsqd_decay_frac,
          "warmup_abs": args.warmup_abs}

    print(f"N = {n:g}, dN = {dn:g}, eta0 = {eta0:g}, eta_inf = {einf:g}, U = {args.U:g}")
    print(f"POW b = {args.pow_b:g}, decay frac = {args.pow_decay_frac:g}; "
          f"WSQD T0 = {args.wsqd_shift:g}, alpha = {args.wsqd_decay_frac:g}; "
          f"absolute warmup = {args.warmup_abs:g}\n")

    # (D1) layer-cake identity, checked by quadrature in the level variable.
    print("check (D1): int T_theta d theta over [eta_inf, eta_0] equals B(N) - N eta_inf")
    print(f"{'sched':<5}{'lhs':>14}{'rhs':>14}{'rel err':>11}")
    levels = einf + (eta0 - einf) * (1.0 - np.cos(np.linspace(0, np.pi / 2, 900))) ** 2
    for name in names:
        tt = np.array([dwell(name, th, n, eta0, einf, **kw) for th in levels])
        lhs = float(np.trapezoid(tt, levels))
        rhs = budget(name, n, eta0, einf, **kw) - n * einf
        print(f"{name:<5}{lhs:>14.6e}{rhs:>14.6e}{abs(lhs - rhs) / abs(rhs):>11.2e}")

    # (D2) the dual conservation law.
    print("\ncheck (D2): int dT_theta/dN d theta equals M(N) + A(N)")
    print(f"{'sched':<5}{'lhs':>14}{'M(N)':>13}{'A(N)':>13}{'M+A':>14}{'rel err':>11}")
    for name in names:
        t1 = np.array([dwell(name, th, n, eta0, einf, **kw) for th in levels])
        t2 = np.array([dwell(name, th, n + dn, eta0, einf, **kw) for th in levels])
        lhs = float(np.trapezoid((t2 - t1) / dn, levels))
        # M(N) = int_0^N d/dN eta_N(t) dt with t HELD FIXED. Rescaling t with the budget
        # would compare a family against itself and return 0 for every rescaling family.
        ts = np.linspace(0.0, n, 20001)
        e1 = eta_of_t(name, ts, n, eta0, einf, **kw)
        e2 = eta_of_t(name, ts, n + dn, eta0, einf, **kw)
        m = float(np.trapezoid((e2 - e1) / dn, ts))
        a = float(eta_of_t(name, np.array([n]), n, eta0, einf, **kw)[0] - einf)
        rel = abs(lhs - (m + a)) / max(abs(m + a), 1e-30)
        print(f"{name:<5}{lhs:>14.6e}{m:>13.5e}{a:>13.5e}{m + a:>14.6e}{rel:>11.2e}")
    print("(M(N) uses fixed t, which is what the time-domain conservation law is stated for;")
    print(" A(N) is the anchoring error at the endpoint)")

    # The design criterion: where in the level spectrum does the sensitivity sit?
    print("\ndwell sensitivity spectrum dT_theta/dN, by level theta/eta_0")
    fracs = [0.9, 0.7, 0.5, 0.3, 0.1, 0.03, 0.01, 0.003]
    print(f"{'sched':<5}" + "".join(f"{f:>9.3g}" for f in fracs)
          + f"{'top half':>10}{'bottom half':>13}")
    for name in names:
        cells, top, bot = [], 0.0, 0.0
        for f in fracs:
            th = f * eta0
            d = (dwell(name, th, n + dn, eta0, einf, **kw)
                 - dwell(name, th, n, eta0, einf, **kw)) / dn
            cells.append(d)
        # split the sensitivity integral at theta = eta_0 / 2
        hi = levels[levels >= eta0 / 2]
        lo = levels[levels < eta0 / 2]
        for arr, acc in ((hi, "top"), (lo, "bot")):
            v = np.array([(dwell(name, th, n + dn, eta0, einf, **kw)
                           - dwell(name, th, n, eta0, einf, **kw)) / dn for th in arr])
            if acc == "top":
                top = float(np.trapezoid(v, arr))
            else:
                bot = float(np.trapezoid(v, arr))
        print(f"{name:<5}" + "".join(f"{c:>9.3f}" for c in cells)
              + f"{top:>10.2e}{bot:>13.2e}")
    print("\ndT_theta/dN near theta = eta_0 is exposure the budget adds at the dangerous end.")
    print("A family is budget-extrapolatable when that number is small, whatever it does lower down.")
    print("\nHow the criterion classifies the published competitors:")
    print("  D2Z lands exactly on L, as it must: linear decay to zero rescales every level.")
    print("  POW and WSQD have a budget-independent base phase, so their high-theta dwell is")
    print("  not merely small but exactly constant in N, and the criterion places them with EH")
    print("  rather than with C/P/L/WSD. They differ from EH in SHAPE, not in the top half: the")
    # Crossover = the base value at the decay onset, the level below which the moving
    # cooldown starts truncating the super-level set. Computed, not quoted, so it stays
    # correct when the shape parameters or N are changed on the command line.
    xo = {name: float(eta_of_t(name, np.array([n - frac * n]), n, eta0, einf, **kw)[0])
          / eta0 for name, frac in (("POW", args.pow_decay_frac),
                                    ("WSQD", args.wsqd_decay_frac))}
    print("  crossover is the base value at the decay onset, eta(N - alpha N) / eta_0, here")
    print(f"  {xo['POW']:.2f} for POW and {xo['WSQD']:.2f} for WSQD: above it sensitivity is 0,")
    print("  below it sensitivity jumps to order 1, whereas EH damps the whole spectrum. So")
    print("  the framework predicts these two survive budget extrapolation, which is exactly")
    print("  why they have to be run as baselines rather than argued away.")


if __name__ == "__main__":
    main()
