import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, Subset
import torchvision
import torchvision.transforms as transforms
import wandb

import polars as pl
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import random
import os
import time


def load_cifar10(subset_ratio=0.1):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    if subset_ratio < 1.0:
        train_size = len(trainset)
        subset_size = int(train_size * subset_ratio)
        train_ics = np.random.choice(train_size, subset_size, replace=False)
        trainset = Subset(trainset, train_ics)

        test_size = len(testset)
        subset_size = int(test_size * subset_ratio)
        test_ics = np.random.choice(test_size, subset_size, replace=False)
        testset = Subset(testset, test_ics)

    return trainset, testset


def load_cifar10_confirmatory(
    split_seed, root="./data", download=True, include_test=True
):
    """Return the frozen 45k/5k/10k CIFAR-10 train, validation, and test views.

    The train and validation subsets use separate dataset objects so random crop and flip can
    never leak into validation. The split is class-stratified and depends only on ``split_seed``.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_view = torchvision.datasets.CIFAR10(
        root=root, train=True, download=download, transform=transform_train
    )
    validation_view = torchvision.datasets.CIFAR10(
        root=root, train=True, download=download, transform=transform_eval
    )
    test = None
    if include_test:
        test = torchvision.datasets.CIFAR10(
            root=root, train=False, download=download, transform=transform_eval
        )

    rng = np.random.default_rng(split_seed)
    labels = np.asarray(train_view.targets)
    train_indices = []
    validation_indices = []
    for label in range(10):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        validation_indices.extend(class_indices[:500].tolist())
        train_indices.extend(class_indices[500:].tolist())
    train_indices.sort()
    validation_indices.sort()
    if len(train_indices) != 45_000 or len(validation_indices) != 5_000:
        raise RuntimeError("unexpected CIFAR-10 class counts while building the frozen split")
    if set(train_indices) & set(validation_indices):
        raise RuntimeError("CIFAR-10 train and validation indices overlap")

    split = {
        "split_seed": int(split_seed),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": list(range(len(test))) if test is not None else [],
    }
    return Subset(train_view, train_indices), Subset(validation_view, validation_indices), test, split


def load_cifar100(subset_ratio=0.1):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    if subset_ratio < 1.0:
        train_size = len(trainset)
        subset_size = int(train_size * subset_ratio)
        train_ics = np.random.choice(train_size, subset_size, replace=False)
        trainset = Subset(trainset, train_ics)

        test_size = len(testset)
        subset_size = int(test_size * subset_ratio)
        test_ics = np.random.choice(test_size, subset_size, replace=False)
        testset = Subset(testset, test_ics)

    return trainset, testset


# Load Data
def load_osc_data(dtype="simple", mode="S", hist=10, pred=10, ratio=0.8):
    """
    dtype:
        - "simple": Simple Harmonic Oscillator
        - "damped": Damped Harmonic Oscillator 

    mode:
        - "S": Single input, Single target ('x')
        - "M": Multi input, Multi target
        - "MS": Multi input, Single target ('x')

    Columns:
        - t: Time (Not used)
        - x: Position
        - v: Velocity
        - a: Acceleration
        - zeta: Damping parameter (0, 0.1, 0.2)
    """
    df = pl.read_parquet("./data/damped_sho.parquet")
    df = df.drop("t")

    # Simple or Damped or Total
    if dtype == "simple":
        df = df.filter(pl.col('zeta') == 0)
    elif dtype == "damped":
        df = df.filter(pl.col('zeta') == 0.01)
    elif dtype == "total":
        df = df
    else:
        raise ValueError("dtype must be 'simple' or 'damped'")

    df = df.select([
        ((pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min())).alias(col)
        for col in df.columns
    ])

    if mode == 'S':
        if dtype == "total":
            print(df)
            df1 = df.filter(pl.col('zeta') == 0)
            df2 = df.filter(pl.col('zeta') == 0.5)
            df3 = df.filter(pl.col('zeta') == 1.0)
            x1  = df1['x'].to_numpy()
            x2  = df2['x'].to_numpy()
            x3  = df3['x'].to_numpy()
            print(x1.shape, x2.shape, x3.shape)
            x1_slide = sliding_window_view(x1, hist + pred)
            x2_slide = sliding_window_view(x2, hist + pred)
            x3_slide = sliding_window_view(x3, hist + pred)
            x1_hist  = x1_slide[:, :hist]
            x2_hist  = x2_slide[:, :hist]
            x3_hist  = x3_slide[:, :hist]
            x1_pred  = x1_slide[:, -pred:]
            x2_pred  = x2_slide[:, -pred:]
            x3_pred  = x3_slide[:, -pred:]
            input1 = torch.tensor(x1_hist, dtype=torch.float32).unsqueeze(2)
            input2 = torch.tensor(x2_hist, dtype=torch.float32).unsqueeze(2)
            input3 = torch.tensor(x3_hist, dtype=torch.float32).unsqueeze(2)
            label1 = torch.tensor(x1_pred, dtype=torch.float32).unsqueeze(2)
            label2 = torch.tensor(x2_pred, dtype=torch.float32).unsqueeze(2)
            label3 = torch.tensor(x3_pred, dtype=torch.float32).unsqueeze(2)
            input_data = torch.cat([input1, input2, input3], dim=0)
            label_data = torch.cat([label1, label2, label3], dim=0)
        else:
            x = df['x'].to_numpy()

            # Sliding window
            x_slide = sliding_window_view(x, hist + pred)   # M x (hist + pred)
            x_hist  = x_slide[:, :hist]                     # M x hist
            x_pred  = x_slide[:, -pred:]                    # M x pred

            input_data = torch.tensor(x_hist, dtype=torch.float32).unsqueeze(2)
            label_data = torch.tensor(x_pred, dtype=torch.float32).unsqueeze(2)
    elif mode == 'MS' or mode == 'M':
        x = df['x'].to_numpy()
        v = df['v'].to_numpy()
        a = df['a'].to_numpy()

        # Sliding window
        x_slide = sliding_window_view(x, hist + pred)   # N x (hist + pred)
        v_slide = sliding_window_view(v, hist + pred)   # N x (hist + pred)
        a_slide = sliding_window_view(a, hist + pred)   # N x (hist + pred)
        x_hist  = x_slide[:, :hist]                     # N x hist
        v_hist  = v_slide[:, :hist]                     # N x hist
        a_hist  = a_slide[:, :hist]                     # N x hist
        x_pred  = x_slide[:, -pred:]                    # N x pred
        v_pred  = v_slide[:, -pred:]                    # N x pred
        a_pred  = a_slide[:, -pred:]                    # N x pred
        x_hist = np.expand_dims(x_hist, axis=2)         # N x hist x 1
        v_hist = np.expand_dims(v_hist, axis=2)         # N x hist x 1
        a_hist = np.expand_dims(a_hist, axis=2)         # N x hist x 1
        
        input_array = np.concatenate([x_hist, v_hist, a_hist], axis=2) # N x hist x 3
        input_data = torch.tensor(input_array, dtype=torch.float32)

        if mode == 'MS':
            label_data = torch.tensor(x_pred, dtype=torch.float32)
            label_data = label_data.unsqueeze(2)
        elif mode == 'M':
            x_pred = np.expand_dims(x_pred, axis=2)
            v_pred = np.expand_dims(v_pred, axis=2)
            a_pred = np.expand_dims(a_pred, axis=2)
            label_array = np.concatenate([x_pred, v_pred, a_pred], axis=2) # N x pred x 3
            label_data = torch.tensor(label_array, dtype=torch.float32)
        else:
            raise ValueError("mode must be 'MS' or 'M'")
    else:
        raise ValueError("mode must be 'S' or 'M'")

    print(f"Input shape: {input_data.shape}")
    print(f"Label shape: {label_data.shape}")

    ds = TensorDataset(input_data, label_data)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    ds_train, ds_val = random_split(ds, [train_size, val_size])
    return ds_train, ds_val


def _load_integral_file(file_path):
    df = pl.read_parquet(file_path)
    return [
        torch.tensor(df[col].to_numpy().reshape(-1, 100), dtype=torch.float32)
        for col in df.columns
    ]


def load_integral(root="data"):
    """Load the legacy train/val operator files without changing their public contract."""
    train_tensors = _load_integral_file(os.path.join(root, "train.parquet"))
    val_tensors = _load_integral_file(os.path.join(root, "val.parquet"))
    return train_tensors, val_tensors


def load_integral_confirmatory(root="data", include_test=False):
    """Load the frozen raw operator splits.

    ``include_test=False`` does not open the test path. HPO callers use that branch so test-set
    isolation is an executable I/O property rather than a convention.
    """
    splits = {
        "train": _load_integral_file(os.path.join(root, "train.parquet")),
        "validation": _load_integral_file(os.path.join(root, "validation.parquet")),
    }
    if include_test:
        splits["test"] = _load_integral_file(os.path.join(root, "test.parquet"))
    return splits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Sharpness measurement ────────────────────────────────────────────────────
#
# "Sharpness" below means the algebraically largest eigenvalue of the Hessian of the training
# loss. That is the definition the edge-of-stability literature uses: Cohen et al.
# (arXiv:2103.00065) section 1 footnote 2, "in this paper, 'sharpness' strictly means the
# maximum eigenvalue of the training loss Hessian", and Gilmer et al. (arXiv:2110.04369)
# section 1 footnote 1, which writes it lambda_1. It is the quantity that the step size has to
# be compared against: Cohen section 2 shows that on the quadratic Taylor approximation, an
# eigenvalue a_i > 2 / eta makes the iterates oscillate along that eigenvector with
# exponentially increasing magnitude, and Damian et al. (arXiv:2209.15594) equation 1 states
# the constrained problem that gradient descent implicitly solves at the edge of stability,
# min_theta L(theta) subject to S(theta) <= 2 / eta.
#
# Three caveats that any use of the recorded numbers has to respect.
#   1. Those results are for full-batch gradient descent. Cohen section 6 states that during
#      SGD the sharpness "does not always settle at any fixed value", and Gilmer section 1
#      states the minibatch version only as the approximate one-sided condition
#      lambda_1 <~ 2 / eta (their section 5 principles 1 and 2).
#   2. This repository trains with AdamW. For an adaptive method the operator that equilibrates
#      is the *preconditioned* Hessian P^{-1} H, and Cohen et al. (arXiv:2207.14484) abstract
#      reports its stability threshold for Adam with beta_1 = 0.9 as 38 / eta, not 2 / eta.
#      `adam_preconditioner` builds that P so the preconditioned sharpness can be recorded
#      next to the raw one instead of reading 2 / eta into a setting it was not derived for.
#   3. The Hessian is evaluated on ONE minibatch, so what is recorded is a minibatch estimate
#      of the training-loss sharpness, not the full-batch quantity Cohen measures.

def _sharpness_grads(loss, params):
    """First derivatives of `loss` with the graph kept, dropping parameters it does not use."""
    params = [p for p in params if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    kept = [(p, g) for p, g in zip(params, grads) if g is not None]
    return [p for p, _ in kept], [g for _, g in kept]


def _dot(xs, ys):
    return float(sum((x * y).sum() for x, y in zip(xs, ys)))


def _power_iteration(hvp, ref, iters, tol, generator, shift=0.0):
    """Dominant-magnitude eigenvalue of the linear operator `hvp`, plus (used_iters, converged).

    `ref` supplies the shapes/dtypes/devices of the vector blocks. The start vector is drawn
    from `generator` on the CPU and moved, so that a CUDA run does not need a CUDA generator and
    the draw stays reproducible across devices. `shift` adds c * v to every product, i.e. it
    runs the iteration on H + c I.
    """
    v = [torch.randn(t.shape, generator=generator, dtype=t.dtype).to(t.device) for t in ref]
    scale = _dot(v, v) ** 0.5
    v = [x / scale for x in v]
    lam = float("nan")
    used = 0
    converged = False
    for i in range(1, int(iters) + 1):
        used = i
        w = hvp(v)
        if shift:
            w = [wi + shift * vi for wi, vi in zip(w, v)]
        # v is a unit vector, so this is the Rayleigh quotient v^T A v.
        new = _dot(v, w)
        norm = max(_dot(w, w), 0.0) ** 0.5
        if norm == 0.0:
            # A v = 0 for a unit v: nothing left for the iteration to refine.
            lam = new
            converged = True
            break
        v = [wi / norm for wi in w]
        if i > 1 and abs(new - lam) <= tol * max(abs(new), 1e-30):
            lam = new
            converged = True
            break
        lam = new
    return lam, used, converged


def hessian_top_eigenvalue(loss, params, iters=20, tol=1e-3, generator=None, precond=None):
    """Largest Hessian eigenvalue of `loss` w.r.t. `params`. Returns (value, used_iters, ok).

    The Hessian is never formed. Every product H v is a double backward: the first
    ``autograd.grad`` keeps the graph, and differentiating grads . v with v held constant gives
    H v exactly, so the cost is one extra backward pass per iteration and the memory cost is one
    retained graph.

    Power iteration converges to the eigenvalue of largest *magnitude*, while the
    edge-of-stability threshold is about the algebraically largest one. If the first pass returns
    mu < 0 then the spectral radius is attained by a negative eigenvalue, every eigenvalue of
    H - mu I is non-negative, and a second pass on that shifted operator returns lambda_max - mu;
    adding mu back recovers lambda_max. That second pass is why `used_iters` can exceed `iters`.

    `precond`, when given, is a {id(param): positive tensor} diagonal P: the value returned is
    then the largest eigenvalue of P^{-1/2} H P^{-1/2}, which is similar to the preconditioned
    Hessian P^{-1} H whose Adam threshold is 38 / eta (arXiv:2207.14484). NaN is returned if any
    parameter is missing from `precond`.
    """
    kept, grads = _sharpness_grads(loss, params)
    if not kept:
        return float("nan"), 0, False
    root = None
    if precond is not None:
        if any(id(p) not in precond for p in kept):
            return float("nan"), 0, False
        root = [precond[id(p)].rsqrt().detach() for p in kept]

    def hvp(vec):
        if root is not None:
            vec = [x * s for x, s in zip(vec, root)]
        out = torch.autograd.grad(grads, kept, grad_outputs=vec, retain_graph=True,
                                  allow_unused=True)
        out = [torch.zeros_like(p) if o is None else o.detach() for o, p in zip(out, kept)]
        if root is not None:
            out = [x * s for x, s in zip(out, root)]
        return out

    mu, used, ok = _power_iteration(hvp, kept, iters, tol, generator)
    if not (mu < 0.0):  # also catches NaN, where a shifted retry would be meaningless
        return mu, used, ok
    nu, used_2, ok_2 = _power_iteration(hvp, kept, iters, tol, generator, shift=-mu)
    return nu + mu, used + used_2, (ok and ok_2)


def adam_preconditioner(optimizer):
    """Diagonal preconditioner P of an Adam-family step, as {id(param): tensor}, or None.

    torch's Adam/AdamW forms the update as lr * m_hat / (sqrt(v_hat) + eps), i.e. -lr P^{-1}
    m_hat with P = diag(sqrt(v_hat) + eps), which is the P in the preconditioned Hessian
    P^{-1} H of arXiv:2207.14484. Returns None while any parameter still has no optimizer state,
    which is the case before the first optimizer step.
    """
    out = {}
    for group in optimizer.param_groups:
        beta_2 = group.get("betas", (0.9, 0.999))[1]
        eps = group.get("eps", 1e-8)
        for p in group["params"]:
            state = optimizer.state.get(p)
            if not state or "exp_avg_sq" not in state:
                return None
            step = state.get("step", 0)
            step = float(step.item()) if torch.is_tensor(step) else float(step)
            bias_2 = 1.0 - beta_2 ** step if step > 0 else 1.0
            out[id(p)] = (state["exp_avg_sq"].detach() / bias_2).sqrt() + eps
    return out


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion=F.mse_loss, acc=False,
                 device="cpu", step_per_batch=False):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = criterion
        self.acc = acc
        self.step_per_batch = step_per_batch
        self.history = []

    def step(self, x):
        pred = self.model(x.to(self.device))
        return pred

    def train_epoch(self, dataloader):
        self.model.train()
        if hasattr(self.optimizer, 'train') and callable(getattr(self.optimizer, 'train')):
            self.optimizer.train()
        epoch_loss = 0
        for x, y in dataloader:
            self.optimizer.zero_grad()
            pred = self.step(x)
            loss = self.criterion(pred, y.to(self.device))
            loss.backward()
            self.optimizer.step()
            if self.step_per_batch and self.scheduler is not None:
                self.scheduler.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        return epoch_loss

    def evaluate(self, dataloader):
        self.model.eval()
        if hasattr(self.optimizer, 'eval') and callable(getattr(self.optimizer, 'eval')):
            self.optimizer.eval()
        eval_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                pred = self.step(x)
                loss = self.criterion(pred, y.to(self.device))
                eval_loss += loss.item()
                if self.acc:
                    _, predicted = torch.max(pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y.to(self.device)).sum().item()
        eval_loss /= len(dataloader)
        if self.acc:
            accuracy = correct / total
            return eval_loss, accuracy
        else:
            return eval_loss

    def train(self, dl_train, dl_val, epochs=500, epoch_callback=None):
        """Train for ``epochs`` epochs.

        ``epoch_callback`` is an optional ``callable(epoch, val_loss)`` invoked at the end of every epoch with the 1-based epoch index. Exceptions raised inside the callback propagate to the caller, which is how an Optuna pruner aborts an unpromising trial mid-training.
        """
        val_loss = 0.0
        accuracy = 0.0
        for epoch in range(epochs):
            train_loss = self.train_epoch(dl_train)
            if self.acc:
                val_loss, accuracy = self.evaluate(dl_val)
            else:
                val_loss = self.evaluate(dl_val)
            if self.scheduler is not None and not self.step_per_batch:
                self.scheduler.step()
            record = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch+1,
                "lr": self.optimizer.param_groups[0]['lr']
            }
            if self.acc:
                record["acc"] = accuracy
            self.history.append(record)
            wandb.log(record)

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1:03}\ttrain_loss={train_loss:.4e}\tval_loss={val_loss:.4e}")

            if epoch_callback is not None:
                epoch_callback(epoch + 1, val_loss)

        if self.acc:
            return val_loss, accuracy
        else:
            return val_loss


class OperatorTrainer:
    def __init__(self, model, optimizer, scheduler, device="cpu", step_per_batch=False,
                 measure_sharpness=0, sharpness_iters=20, sharpness_tol=1e-3,
                 sharpness_precond=False, sharpness_seed=0):
        """``measure_sharpness=0`` (the default) is exactly the pre-2026-08-17 trainer.

        A positive k measures the sharpness of one frozen training batch every k epochs and adds
        the sharpness columns to ``history``. The measurement is read-only with respect to
        training: it runs after the last optimizer step of the epoch, uses ``autograd.grad``
        (which never writes ``.grad``), touches no optimizer state, restores the module's
        training flag, and draws its power-iteration start vector from a private generator so
        that the global RNG stream feeding the dataloader shuffle stays untouched. Switching
        measurement on therefore leaves the loss curve bit-identical.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.step_per_batch = step_per_batch
        self.history = []
        self.measure_sharpness = int(measure_sharpness)
        self.sharpness_iters = int(sharpness_iters)
        self.sharpness_tol = float(sharpness_tol)
        self.sharpness_precond = bool(sharpness_precond)
        self._sharpness_batch = None
        self._sharpness_gen = None
        if self.measure_sharpness:
            self._sharpness_gen = torch.Generator().manual_seed(sharpness_seed)

    def step(self, u, y):
        pred = self.model(u.to(self.device), y.to(self.device))
        return pred

    def train_epoch(self, dataloader):
        self.model.train()
        if hasattr(self.optimizer, 'train') and callable(getattr(self.optimizer, 'train')):
            self.optimizer.train()
        epoch_loss = 0
        for u, y, Guy in dataloader:
            if self.measure_sharpness and self._sharpness_batch is None:
                # Freeze the first training batch ever seen and reuse it for every later
                # measurement, so the sharpness trajectory is read off one fixed loss surface
                # instead of a different random surface each time. Captured from inside the
                # training loop rather than by iterating the dataloader a second time, because
                # a second iterator would draw a seed from the global RNG and change the
                # shuffle order of training itself.
                self._sharpness_batch = tuple(t.detach().clone() for t in (u, y, Guy))
            self.optimizer.zero_grad()
            pred = self.step(u, y)
            loss = F.mse_loss(pred, Guy.to(self.device))
            loss.backward()
            self.optimizer.step()
            if self.step_per_batch and self.scheduler is not None:
                self.scheduler.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataloader)
        return epoch_loss

    def evaluate(self, dataloader):
        self.model.eval()
        if hasattr(self.optimizer, 'eval') and callable(getattr(self.optimizer, 'eval')):
            self.optimizer.eval()
        eval_loss = 0
        with torch.no_grad():
            for u, y, Guy in dataloader:
                pred = self.step(u, y)
                loss = F.mse_loss(pred, Guy.to(self.device))
                eval_loss += loss.item()
        eval_loss /= len(dataloader)
        return eval_loss

    def sharpness(self):
        """Sharpness of the frozen batch at the current parameters: (raw, precond, iters, ok).

        ``precond`` is NaN unless ``sharpness_precond`` was requested. The forward pass runs in
        eval mode so that the Hessian is a deterministic function of the parameters; for the two
        operator models here (DeepONet has no stochastic layer, TFONet defaults to dropout 0.0)
        eval and train mode compute the same function, so this costs no fidelity.
        """
        u, y, Guy = self._sharpness_batch
        was_training = self.model.training
        self.model.eval()
        try:
            loss = F.mse_loss(self.step(u, y), Guy.to(self.device))
            raw, iters, ok = hessian_top_eigenvalue(
                loss, self.model.parameters(), iters=self.sharpness_iters,
                tol=self.sharpness_tol, generator=self._sharpness_gen)
            pre = float("nan")
            if self.sharpness_precond:
                pre, iters_pre, ok_pre = hessian_top_eigenvalue(
                    loss, self.model.parameters(), iters=self.sharpness_iters,
                    tol=self.sharpness_tol, generator=self._sharpness_gen,
                    precond=adam_preconditioner(self.optimizer))
                iters += iters_pre
                ok = ok and ok_pre
        finally:
            if was_training:
                self.model.train()
        return raw, pre, iters, ok

    def _sharpness_record(self, epoch):
        """The sharpness columns of one history row. Empty dict when measurement is off.

        Called after the epoch's last optimizer step and before the epoch-level
        ``scheduler.step()``, so ``lr_pre_step`` is the step size that was in force during this
        epoch, which is what has to be compared against 2 / lambda_max. The ``lr`` column cannot
        be used for that comparison: it is read after ``scheduler.step()`` and therefore holds
        next epoch's value. For a per-batch scheduler ``lr_pre_step`` is the value after this
        epoch's final batch, since the step size varied within the epoch.
        """
        if not self.measure_sharpness:
            return {}
        out = {"lambda_max": float("nan"), "sharpness_iters": 0,
               "sharpness_converged": False, "sharpness_secs": 0.0,
               "lr_pre_step": self.optimizer.param_groups[0]['lr']}
        if self.sharpness_precond:
            out["lambda_max_precond"] = float("nan")
        if epoch % self.measure_sharpness or self._sharpness_batch is None:
            return out
        started = time.perf_counter()
        raw, pre, iters, ok = self.sharpness()
        out["lambda_max"] = raw
        out["sharpness_iters"] = iters
        out["sharpness_converged"] = ok
        out["sharpness_secs"] = time.perf_counter() - started
        if self.sharpness_precond:
            out["lambda_max_precond"] = pre
        return out

    def train(self, dl_train, dl_val, epochs=500, epoch_callback=None):
        """Train for ``epochs`` epochs.

        ``epoch_callback`` follows the same contract as :meth:`Trainer.train`: it is called as ``epoch_callback(epoch, val_loss)`` at the end of every epoch and its exceptions propagate.
        """
        val_loss = 0.0
        for epoch in range(epochs):
            train_loss = self.train_epoch(dl_train)
            val_loss = self.evaluate(dl_val)
            sharp = self._sharpness_record(epoch + 1)
            if self.scheduler is not None and not self.step_per_batch:
                self.scheduler.step()
            record = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch+1,
                "lr": self.optimizer.param_groups[0]['lr']
            }
            record.update(sharp)
            self.history.append(record)
            wandb.log(record)
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1:03}\ttrain_loss={train_loss:.4e}\tval_loss={val_loss:.4e}")
            if epoch_callback is not None:
                epoch_callback(epoch + 1, val_loss)
        return val_loss
