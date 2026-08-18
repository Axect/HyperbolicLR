import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from hyperbolic_lr import HyperbolicLR, ExpHyperbolicLR


# ── Scheduler Registry ──────────────────────────────────────────────────────

def get_schedulers():
    return {
        "N": None,
        "P": lr_scheduler.PolynomialLR,
        "C": lr_scheduler.CosineAnnealingLR,
        "E": lr_scheduler.ExponentialLR,
        "H": HyperbolicLR,
        "EH": ExpHyperbolicLR,
        "L": lr_scheduler.LinearLR,
        "S": lr_scheduler.StepLR,
        "OC": lr_scheduler.OneCycleLR,
        "CY": lr_scheduler.CyclicLR,
        "WH": HyperbolicLR,
        "WEH": ExpHyperbolicLR,
        "WC": "_factory",  # handled by create_scheduler
        "WSD": "_factory",  # handled by create_scheduler
        # Competitor baselines for the budget-extrapolation study. POW and WSQD
        # are the two published schedulers whose base phase is deliberately
        # budget-independent; D2Z is the maximally budget-dependent control.
        "POW": "_factory",  # handled by create_scheduler
        "WSQD": "_factory",  # handled by create_scheduler
        "D2Z": lr_scheduler.LinearLR,
    }


def get_scheduler_names():
    return list(get_schedulers().keys())


STEP_PER_BATCH_SCHEDULERS = {"OC", "CY"}


def is_step_per_batch(scheduler_name):
    return scheduler_name in STEP_PER_BATCH_SCHEDULERS


# ── Competitor Baseline Constants ────────────────────────────────────────────
#
# POW: Power scheduler, Shen et al., arXiv:2408.13359.
#   Eq. (7): eta_power(n) = min(eta_max, beta * a * n^b), n = tokens seen so far.
#   Eq. (8): Power(n) = (n / N_warmup) * eta_power(N_warmup)      if n < N_warmup
#                     = eta_power(n)                              if N_warmup < n <= N - N_decay
#                     = f(n, N, N_decay) * eta_power(N - N_decay)  if n > N - N_decay
#   Section 4: "we select a = 4 and b = -0.51 for the rest of this paper", and
#   eta_max "has a very limited impact on the overall learning rate curve, we can
#   simply set it to a large enough value" (0.02 there).
#   Section 3, Tables 2 and 3: warmup is 1B tokens FIXED while the token budget T
#   sweeps 2B to 256B, and the decay phase is 10% of tokens. So both the warmup
#   length and the base curve are budget-independent by construction; only the
#   decay onset moves with the budget.
#
# WSQD: Ma and Chen, arXiv:2607.10959.
#   Eq. (3), post-warmup: eta_t = c0 / sqrt(t + T0)                      if 1 <= t <= (1 - alpha) T
#                              = c0 / sqrt((1 - alpha) T + T0) * (T - t) / (alpha T)
#                                                                        if (1 - alpha) T < t <= T
#   Section 3.1: "We use the same 300-step warmup and a final linear decay phase
#   for all schedules with decay fraction alpha = 0.2." Theorem 1 requires
#   alpha in (0, 1/2). Section 3.4: "we recommend the simple default T0 = T1,
#   setting the shift equal to the length of the initial pilot horizon."
#
# Both papers state the LR in absolute units (c0, or beta * a). This project's
# tuned quantity is the PEAK lr, so both factories below are written as a
# multiplicative factor that equals 1 at the end of warmup. For POW that is the
# identification eta_max = lr and beta * a = lr * warmup^(-b), i.e. the min() in
# Eq. (7) binds exactly on [0, warmup]; for WSQD it is c0 = lr * sqrt(warmup + T0).
# The shape parameters (b, alpha, T0) are untouched by this reparameterization.
POW_B = -0.51
POW_DECAY_FRAC = 0.1
WSQD_DECAY_FRAC = 0.2
# T0 = the pilot horizon, per Section 3.4's rule T0 = T1. This project's pilot is the
# HPO budget, 10 epochs (phase2_orchestrate_extrap.py TUNE_BUDGETS starts at 10).
WSQD_SHIFT = 10.0
# Absolute warmup length in epochs, NOT a fraction of the budget. Both papers fix the
# warmup in absolute units across horizons (1B tokens while T sweeps 2B to 256B; 300
# steps while T sweeps 15k to 60k), and a budget-proportional warmup would drag the
# whole base curve along with the budget and destroy the one property these two
# schedulers were designed to have. The value has to fit inside the smallest budget
# this project tunes at (10 epochs) and still leave room for a base phase and a decay
# phase, so 5: half of that smallest budget, which is the same warmup-to-horizon ratio
# the Power scheduler uses at its own smallest token budget (1B of 2B), and 5% of the
# reference budget N = 100. Budgets of 5 epochs or fewer degenerate, since the ramp
# then fills the whole run and the base anchor gets clamped with it.
ABS_WARMUP_EPOCHS = 5


# ── Scheduler Factory ────────────────────────────────────────────────────────

def create_scheduler(scheduler_name, optimizer, lr, infimum_lr, epochs,
                     steps_per_epoch=1, **extra_params):
    if scheduler_name == "N":
        return None

    params = get_scheduler_params(
        scheduler_name, lr, infimum_lr, epochs, steps_per_epoch, **extra_params
    )

    if scheduler_name == "WC":
        return _create_warmup_cosine(optimizer, lr, infimum_lr, epochs)

    if scheduler_name == "WSD":
        cooldown_frac = extra_params.get("cooldown_frac", 0.2)
        return _create_wsd(optimizer, lr, infimum_lr, epochs,
                           cooldown_frac=cooldown_frac)

    if scheduler_name == "POW":
        return _create_power(
            optimizer, epochs,
            pow_b=extra_params.get("pow_b", POW_B),
            warmup_epochs=extra_params.get("warmup_epochs", ABS_WARMUP_EPOCHS),
            pow_decay_frac=extra_params.get("pow_decay_frac", POW_DECAY_FRAC),
        )

    if scheduler_name == "WSQD":
        return _create_wsqd(
            optimizer, epochs,
            decay_frac=extra_params.get("decay_frac", WSQD_DECAY_FRAC),
            shift=extra_params.get("shift", WSQD_SHIFT),
            warmup_epochs=extra_params.get("warmup_epochs", ABS_WARMUP_EPOCHS),
        )

    cls = get_schedulers()[scheduler_name]
    return cls(optimizer, **params)


def _create_warmup_cosine(optimizer, lr, infimum_lr, epochs):
    warmup_epochs = max(1, epochs // 10)
    warmup = lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=infimum_lr
    )
    return lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs]
    )


def _create_wsd(optimizer, lr, infimum_lr, epochs, cooldown_frac=0.2):
    warmup_epochs = max(1, epochs // 10)
    cooldown_epochs = max(1, int(epochs * cooldown_frac))
    # Guard: warmup + cooldown must not exceed epochs
    if warmup_epochs + cooldown_epochs > epochs:
        cooldown_epochs = max(0, epochs - warmup_epochs)
    if warmup_epochs > epochs:
        warmup_epochs = epochs
    min_factor = max(infimum_lr / lr if lr > 0 else 1e-6, 1e-6)
    cooldown_start = epochs - cooldown_epochs

    def wsd_lambda(t):
        if t < warmup_epochs:
            # linear warmup: start at 0.01, reach 1.0 at warmup_epochs
            if warmup_epochs <= 1:
                progress = 1.0
            else:
                progress = t / (warmup_epochs - 1) if warmup_epochs > 1 else 1.0
            factor = 0.01 + 0.99 * progress
        elif t < cooldown_start or cooldown_epochs == 0:
            factor = 1.0
        else:
            progress = (t - cooldown_start) / cooldown_epochs
            progress = min(max(progress, 0.0), 1.0)
            factor = min_factor + (1.0 - min_factor) * (1.0 - progress ** 0.5)
        return max(factor, 1e-9)

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=wsd_lambda)


def _create_power(optimizer, epochs, pow_b=POW_B,
                  warmup_epochs=ABS_WARMUP_EPOCHS,
                  pow_decay_frac=POW_DECAY_FRAC):
    """Power scheduler of Shen et al., arXiv:2408.13359, Eq. (7) and Eq. (8).

    ``epochs`` is used ONLY to place the decay onset at ``(1 - pow_decay_frac) * epochs``,
    exactly as Eq. (8) uses N only in its third branch. The warmup length and the
    whole base phase ``eta_power(t) = min(eta_max, beta * a * t^b)`` are independent of
    ``epochs``, which is the property the paper's title claims ("token number agnostic")
    and the property this project measures. So POW's initial segment is bit-identical
    across budgets, and only the tail moves.

    Eq. (8) leaves the decay function f(n, N, N_decay) as whatever WSD uses; the
    project's WSD (``_create_wsd``) uses the 1 - sqrt shape of Hagele et al., so POW
    reuses it here to keep the two comparable. Unlike WSD it anneals to 0 rather than
    to ``infimum_lr``, following the description of the WSD decay phase in Section 1
    of the same paper ("annealing the learning rate to 0"). ``infimum_lr`` therefore
    plays no role for this scheduler.
    """
    # Clamp the ramp before sizing the decay, so a budget shorter than the absolute
    # warmup cannot leave warmup past the end of the run.
    warmup = min(max(1, int(warmup_epochs)), max(1, epochs))
    decay_epochs = max(1, int(epochs * pow_decay_frac))
    if warmup + decay_epochs > epochs:
        decay_epochs = max(0, epochs - warmup)
    decay_start = epochs - decay_epochs

    def base(t):
        # min(eta_max, beta * a * t^b) / eta_max with eta_max = lr and
        # beta * a = lr * warmup^(-b): the cap binds exactly on [0, warmup].
        return (max(float(t), float(warmup)) / warmup) ** pow_b

    def power_lambda(t):
        if t < warmup:
            # Eq. (8) first branch, ramping to eta_power(N_warmup) = lr at t = warmup.
            # Starts at 0.01 rather than 0 so that epoch 0 is not a dead epoch, which
            # is the same convention as _create_warmup_cosine and _create_wsd.
            factor = 0.01 + 0.99 * (t / warmup)
        elif t <= decay_start or decay_epochs == 0:
            factor = base(t)
        else:
            progress = min(max((t - decay_start) / decay_epochs, 0.0), 1.0)
            factor = base(decay_start) * (1.0 - progress ** 0.5)
        return max(factor, 1e-9)

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=power_lambda)


def _create_wsqd(optimizer, epochs, decay_frac=WSQD_DECAY_FRAC, shift=WSQD_SHIFT,
                 warmup_epochs=ABS_WARMUP_EPOCHS):
    """WSqD of Ma and Chen, arXiv:2607.10959, Eq. (3), plus the paper's fixed warmup.

    The base phase c0 / sqrt(t + T0) is horizon-free: ``epochs`` enters only through
    the onset ``(1 - alpha) * epochs`` of the final linear cooldown and through the
    cooldown slope, which is what the paper means by "WSqD introduces horizon
    dependence only during the final cooldown". Written multiplicatively with
    c0 = lr * sqrt(warmup + T0), so the factor is 1 at the end of warmup and the base
    values at fixed t do not depend on ``epochs``.

    Eq. (3)'s cooldown is linear and reaches 0 at t = T, so ``infimum_lr`` plays no
    role here either. The 1e-9 floor is the same guard ``_create_wsd`` uses.
    """
    warmup = min(max(1, int(warmup_epochs)), max(1, epochs))
    decay_epochs = max(1, int(epochs * decay_frac))
    if warmup + decay_epochs > epochs:
        decay_epochs = max(0, epochs - warmup)
    decay_start = epochs - decay_epochs
    shift = max(float(shift), 0.0)

    def base(t):
        return ((warmup + shift) / (float(t) + shift)) ** 0.5

    def wsqd_lambda(t):
        if t < warmup:
            factor = 0.01 + 0.99 * (t / warmup)
        elif t <= decay_start or decay_epochs == 0:
            factor = base(t)
        else:
            # Eq. (3) second branch: (T - t) / (alpha T), with alpha T = decay_epochs.
            factor = base(decay_start) * (epochs - t) / decay_epochs
        return max(factor, 1e-9)

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=wsqd_lambda)


# ── Default Scheduler Parameters ─────────────────────────────────────────────

def get_scheduler_params(name, lr, infimum_lr, epochs, steps_per_epoch=1,
                         **extra_params):
    warmup_epochs = max(1, epochs // 10)
    total_steps = epochs * steps_per_epoch

    p = {
        "N": {},
        "P": {"power": 0.5, "total_iters": epochs},
        "C": {"T_max": epochs, "eta_min": infimum_lr},
        "E": {"gamma": 0.96},
        "H": {
            "upper_bound": 250,
            "max_iter": epochs,
            "infimum_lr": infimum_lr,
        },
        "EH": {
            "upper_bound": 250,
            "max_iter": epochs,
            "infimum_lr": infimum_lr,
        },
        "L": {
            "start_factor": 1.0,
            "end_factor": max(infimum_lr / lr, 1e-6),
            "total_iters": epochs,
        },
        "S": {
            "step_size": max(1, epochs // 3),
            "gamma": 0.1,
        },
        "OC": {
            "max_lr": lr,
            "total_steps": total_steps,
            "pct_start": 0.3,
        },
        "CY": {
            "base_lr": infimum_lr,
            "max_lr": lr,
            "step_size_up": max(1, total_steps // 4),
        },
        "WH": {
            "upper_bound": 250,
            "max_iter": epochs,
            "infimum_lr": infimum_lr,
            "warmup_epochs": warmup_epochs,
        },
        "WEH": {
            "upper_bound": 250,
            "max_iter": epochs,
            "infimum_lr": infimum_lr,
            "warmup_epochs": warmup_epochs,
        },
        "WC": {},  # handled by factory
        "WSD": {},  # handled by factory
        "POW": {},  # handled by factory
        "WSQD": {},  # handled by factory
        # D2Z: linear decay to zero. Distinct from "L", which stops at
        # infimum_lr / lr; here the terminal factor is exactly 0, which torch
        # accepts (LinearLR rejects only end_factor < 0). The last TRAINING epoch
        # still runs at lr / epochs, because the epoch loop steps after training,
        # so the exact zero is only ever reached by the post-loop step and shows up
        # only in the recorded lr column of the final row.
        "D2Z": {
            "start_factor": 1.0,
            "end_factor": 0.0,
            "total_iters": epochs,
        },
    }

    params = p.get(name, {}).copy()
    params.update(extra_params)
    return params


# ── Optuna Search Space ──────────────────────────────────────────────────────

def get_optuna_search_space(trial, scheduler_name, lr, epochs,
                            steps_per_epoch=1):
    hparams = {}
    total_steps = epochs * steps_per_epoch

    if scheduler_name == "N":
        pass
    elif scheduler_name == "P":
        hparams["power"] = trial.suggest_float("power", 0.5, 3.0)
    elif scheduler_name == "E":
        hparams["gamma"] = trial.suggest_float("gamma", 0.9, 0.99)
    elif scheduler_name in ("C", "H", "EH", "WH", "WEH", "WC", "WSD"):
        hparams["infimum_lr"] = trial.suggest_float(
            "infimum_lr", 1e-7, 1e-4, log=True
        )
        if scheduler_name in ("H", "EH", "WH", "WEH"):
            hparams["upper_bound"] = trial.suggest_int(
                "upper_bound", 200, 400, step=50
            )
        if scheduler_name == "WSD":
            hparams["cooldown_frac"] = trial.suggest_float(
                "cooldown_frac", 0.1, 0.4
            )
    elif scheduler_name == "L":
        hparams["end_factor"] = trial.suggest_float(
            "end_factor", 0.001, 0.1, log=True
        )
    elif scheduler_name == "S":
        hparams["step_size"] = trial.suggest_int(
            "step_size", max(1, epochs // 10), max(2, epochs // 2)
        )
        hparams["gamma"] = trial.suggest_float("gamma", 0.05, 0.5)
    elif scheduler_name == "OC":
        hparams["pct_start"] = trial.suggest_float("pct_start", 0.1, 0.5)
    elif scheduler_name == "CY":
        hparams["step_size_up"] = trial.suggest_int(
            "step_size_up", steps_per_epoch, max(steps_per_epoch + 1, total_steps // 2)
        )
    elif scheduler_name == "POW":
        # arXiv:2408.13359 Table 3 searches b over [-0.6, -0.4] and reports the
        # optimum consistently inside [-0.53, -0.49]. a is not searched here because
        # it is absorbed into the peak lr, which the caller already tunes, and
        # eta_max is pinned to that same peak. pow_decay_frac stays at the paper's
        # fixed 10% so POW carries the same number of tuned dimensions as WSD.
        hparams["pow_b"] = trial.suggest_float("pow_b", -0.6, -0.4)
    elif scheduler_name == "WSQD":
        # arXiv:2607.10959 fixes alpha = 0.2 in every experiment and Theorem 1 needs
        # alpha in (0, 1/2). The range below is the project's own WSD cooldown_frac
        # range, which sits inside that interval and brackets the paper's default, so
        # WSQD and WSD are tuned over the same cooldown budget. The shift T0 is not
        # searched: Section 3.4 gives the rule T0 = T1 (the pilot horizon) instead,
        # and reports insensitivity once T0 is moderately large.
        hparams["decay_frac"] = trial.suggest_float("decay_frac", 0.1, 0.4)
    elif scheduler_name == "D2Z":
        pass  # linear decay to zero has no shape parameter beyond the peak lr

    return hparams


def build_scheduler_from_trial(scheduler_name, optimizer, trial_params, lr,
                               epochs, steps_per_epoch=1):
    infimum_lr = trial_params.get("infimum_lr", 1e-6)
    extra = {}

    if scheduler_name == "P":
        extra["power"] = trial_params["power"]
    elif scheduler_name == "E":
        extra["gamma"] = trial_params["gamma"]
    elif scheduler_name in ("C", "WC"):
        pass  # infimum_lr is enough
    elif scheduler_name == "WSD":
        extra["cooldown_frac"] = trial_params.get("cooldown_frac", 0.2)
    elif scheduler_name in ("H", "EH", "WH", "WEH"):
        extra["upper_bound"] = trial_params["upper_bound"]
    elif scheduler_name == "L":
        extra["end_factor"] = trial_params["end_factor"]
        extra["start_factor"] = 1.0
        extra["total_iters"] = epochs
    elif scheduler_name == "S":
        extra["step_size"] = trial_params["step_size"]
        extra["gamma"] = trial_params["gamma"]
    elif scheduler_name == "OC":
        extra["pct_start"] = trial_params["pct_start"]
    elif scheduler_name == "CY":
        extra["step_size_up"] = trial_params["step_size_up"]
    elif scheduler_name == "POW":
        extra["pow_b"] = trial_params.get("pow_b", POW_B)
    elif scheduler_name == "WSQD":
        extra["decay_frac"] = trial_params.get("decay_frac", WSQD_DECAY_FRAC)
    elif scheduler_name == "D2Z":
        pass  # shape comes entirely from get_scheduler_params at the deploy budget

    return create_scheduler(
        scheduler_name, optimizer, lr, infimum_lr, epochs,
        steps_per_epoch=steps_per_epoch, **extra
    )


# ── Optimize-Compare Parameter Extraction ─────────────────────────────────────

def get_optimize_compare_params(params, scheduler_name, lr, epochs,
                                steps_per_epoch=1):
    infimum_lr = params.get("infimum_lr", 1e-6)
    extra = {}

    if scheduler_name == "N":
        return None, {}
    elif scheduler_name == "P":
        extra["power"] = params["power"]
        extra["total_iters"] = epochs
    elif scheduler_name == "E":
        extra["gamma"] = params["gamma"]
    elif scheduler_name in ("C", "WC"):
        pass
    elif scheduler_name == "WSD":
        extra["cooldown_frac"] = params.get("cooldown_frac", 0.2)
    elif scheduler_name in ("H", "EH", "WH", "WEH"):
        extra["upper_bound"] = params["upper_bound"]
    elif scheduler_name == "L":
        extra["end_factor"] = params.get("end_factor", infimum_lr / lr)
        extra["start_factor"] = 1.0
        extra["total_iters"] = epochs
    elif scheduler_name == "S":
        extra["step_size"] = params["step_size"]
        extra["gamma"] = params["gamma"]
    elif scheduler_name == "OC":
        extra["pct_start"] = params.get("pct_start", 0.3)
    elif scheduler_name == "CY":
        extra["step_size_up"] = params.get("step_size_up", max(1, epochs * steps_per_epoch // 4))
    elif scheduler_name == "POW":
        extra["pow_b"] = params.get("pow_b", POW_B)
    elif scheduler_name == "WSQD":
        extra["decay_frac"] = params.get("decay_frac", WSQD_DECAY_FRAC)
    elif scheduler_name == "D2Z":
        # total_iters is re-derived from the DEPLOY budget by get_scheduler_params,
        # which is the whole point of D2Z as the rescaling control.
        pass

    return infimum_lr, extra


# ── LR-Free Optimizer Registry ───────────────────────────────────────────────

def get_lr_free_optimizers():
    opts = {}
    try:
        from prodigyopt import Prodigy
        opts["Prodigy"] = Prodigy
    except ImportError:
        pass
    try:
        from dadaptation import DAdaptAdam
        opts["DAdapt"] = DAdaptAdam
    except ImportError:
        pass
    try:
        from schedulefree import AdamWScheduleFree
        opts["ScheduleFree"] = AdamWScheduleFree
    except ImportError:
        pass
    return opts


def get_lr_free_default_lr(optimizer_name):
    if optimizer_name in ("Prodigy", "DAdapt"):
        return 1.0
    if optimizer_name == "ScheduleFree":
        return 1e-3
    return 1.0


def get_lr_free_optuna_search_space(trial, optimizer_name):
    hparams = {}
    if optimizer_name == "Prodigy":
        hparams["d_coef"] = trial.suggest_float("d_coef", 0.1, 10.0, log=True)
    elif optimizer_name == "DAdapt":
        hparams["growth_rate"] = trial.suggest_float("growth_rate", 1.01, 2.0)
    elif optimizer_name == "ScheduleFree":
        hparams["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hparams["warmup_steps"] = trial.suggest_int("warmup_steps", 0, 500)
    return hparams
