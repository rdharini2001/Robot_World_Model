"""Learned residual observers.

A learned observer = base integrator (dead reckoning or EKF) + a recurrent
network that outputs a residual correction.  Because the readout is
initialized near zero, an untrained learned observer == its base prior.

Cell in {"gru","ssm"}.  Anchor in {"dr","ekf"}.
"""
try:
    import autograd.numpy as anp
except ImportError:  # inference-only fallback for environments without autograd
    import numpy as anp
import numpy as np
from . import nn_core as nc
from . import ekf as ekf_mod
from . import dynamics as dyn


def d_of(params):
    return params["Wo"].shape[1]


def _cell(cellname, params, x, h, selective):
    if cellname == "gru":
        return nc.gru_cell(params, x, h)
    return nc.ssm_cell(params, x, h, selective=selective)


# ---- batched, differentiable sequence forward (training) --------------------
def run_seq(params, X, anchors, cellname, selective):
    """X:(B,T,F) anchors:(B,T,3) -> est:(B,T,3) = anchor + residual."""
    B, T, F = X.shape
    d = d_of(params)
    h = anp.zeros((B, d))
    outs = []
    for t in range(T):
        h, y = _cell(cellname, params, X[:, t, :], h, selective)
        outs.append(anchors[:, t, :] + y)
    return anp.stack(outs, axis=1)


def seq_loss(params, X, anchors, targets, cellname, selective, lam=0.5):
    est = run_seq(params, X, anchors, cellname, selective)
    dpos = est[:, :, :2] - targets[:, :, :2]
    lpos = anp.mean(anp.sum(dpos ** 2, axis=-1))
    lhead = anp.mean(1.0 - anp.cos(est[:, :, 2] - targets[:, :, 2]))
    return lpos + lam * lhead


# ---- online observer (closed loop) ------------------------------------------
class LearnedObserver:
    def __init__(self, params, cellname, anchor, params_sim, landmarks,
                 use_mask=True, selective=True, use_cov=False):
        self.p = params
        self.cellname = cellname
        self.selective = selective
        self.use_mask = use_mask
        self.use_cov = use_cov
        self.dt = params_sim.dt
        self.anchor_kind = anchor
        if anchor == "ekf":
            self.base = ekf_mod.EKF(params_sim, landmarks)
        else:
            self.base = ekf_mod.DeadReckoning(params_sim)

    def reset(self, s0):
        self.base.reset(s0)
        self.h = np.zeros(d_of(self.p))

    def step(self, u_odo, z, mask, k):
        anchor = self.base.step(u_odo, z, mask, k)      # base advances
        Pdiag = np.diag(self.base.P).copy() if (self.use_cov and self.anchor_kind == "ekf") else None
        x = nc.featurize_step(u_odo, z, mask, anchor, self.dt,
                              use_mask=self.use_mask, Pdiag=Pdiag)
        self.h, y = _cell(self.cellname, self.p, x, self.h, self.selective)
        est = anchor + np.asarray(y)
        est[2] = dyn.wrap(est[2])
        return est
