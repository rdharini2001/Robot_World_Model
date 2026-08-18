"""Train a learned residual observer by BPTT (autograd) on logged trajectories."""
from dataclasses import dataclass
import time
import numpy as np
try:
    from autograd import grad
except ImportError:
    grad = None
from . import data as data_mod, nn_core as nc, models


@dataclass
class Spec:
    name: str
    cell: str = "ssm"          # "gru" | "ssm"
    anchor: str = "dr"         # "dr"  | "ekf"
    selective: bool = True
    use_mask: bool = True
    use_cov: bool = False      # feed EKF covariance diag (ekf anchor only)
    d: int = 32
    lr: float = 3e-3
    epochs: int = 25
    batch: int = 24
    lam: float = 0.5
    seed: int = 0


def _init(spec, F, rng):
    if spec.cell == "gru":
        return nc.init_gru(F, spec.d, rng)
    return nc.init_ssm(F, spec.d, rng, selective=spec.selective)


def train(spec, dataset, val_dataset=None, verbose=True):
    if grad is None:
        raise RuntimeError("Training requires autograd. Existing checkpoints and all evaluations run without it.")
    rng = np.random.default_rng(spec.seed)
    X, A, Y = data_mod.featurize(dataset, spec.anchor, spec.use_mask, spec.use_cov)
    N, T, F = X.shape
    params = _init(spec, F, rng)
    opt = nc.Adam(list(params.keys()), lr=spec.lr)
    gfun = grad(lambda p, xb, ab, yb: models.seq_loss(
        p, xb, ab, yb, spec.cell, spec.selective, spec.lam))
    lfun = lambda p, xb, ab, yb: float(models.seq_loss(
        p, xb, ab, yb, spec.cell, spec.selective, spec.lam))

    if val_dataset is not None:
        Xv, Av, Yv = data_mod.featurize(val_dataset, spec.anchor, spec.use_mask, spec.use_cov)

    best = (1e9, {k: v.copy() for k, v in params.items()})
    t0 = time.time()
    for ep in range(spec.epochs):
        idx = rng.permutation(N)
        tot = 0.0
        for b in range(0, N, spec.batch):
            bi = idx[b:b + spec.batch]
            g = gfun(params, X[bi], A[bi], Y[bi])
            params = opt.step(params, g)
            tot += lfun(params, X[bi], A[bi], Y[bi]) * len(bi)
        tr = tot / N
        if val_dataset is not None:
            vl = lfun(params, Xv, Av, Yv)
            if vl < best[0]:
                best = (vl, {k: v.copy() for k, v in params.items()})
        else:
            vl = tr
            best = (tr, {k: v.copy() for k, v in params.items()})
        if verbose and (ep % 5 == 0 or ep == spec.epochs - 1):
            print(f"  [{spec.name}] ep{ep:02d} train={tr:.4f} val={vl:.4f} "
                  f"({time.time()-t0:.0f}s)")
    return best[1]
