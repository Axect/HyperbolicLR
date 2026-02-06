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
    }


def get_scheduler_names():
    return list(get_schedulers().keys())


STEP_PER_BATCH_SCHEDULERS = {"OC", "CY"}


def is_step_per_batch(scheduler_name):
    return scheduler_name in STEP_PER_BATCH_SCHEDULERS


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
    elif scheduler_name in ("C", "H", "EH", "WH", "WEH", "WC"):
        hparams["infimum_lr"] = trial.suggest_float(
            "infimum_lr", 1e-7, 1e-4, log=True
        )
        if scheduler_name in ("H", "EH", "WH", "WEH"):
            hparams["upper_bound"] = trial.suggest_int(
                "upper_bound", 200, 400, step=50
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

    return infimum_lr, extra


# ── LR-Free Optimizer Registry ───────────────────────────────────────────────

def get_lr_free_optimizers():
    try:
        from prodigyopt import Prodigy
        from dadaptation import DAdaptAdam
        return {
            "Prodigy": Prodigy,
            "DAdapt": DAdaptAdam,
        }
    except ImportError:
        return {}


def get_lr_free_optuna_search_space(trial, optimizer_name):
    hparams = {}
    if optimizer_name == "Prodigy":
        hparams["d_coef"] = trial.suggest_float("d_coef", 0.1, 10.0, log=True)
    elif optimizer_name == "DAdapt":
        hparams["growth_rate"] = trial.suggest_float("growth_rate", 1.01, 2.0)
    return hparams
