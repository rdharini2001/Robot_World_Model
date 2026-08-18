"""Autograd-based neural core: featurization, parameter init, Adam, and the
two recurrent cells (GRU and a compact selective diagonal SSM).

Cells are written so the SAME code runs (a) batched + differentiable during
training (x has shape (B,F)) and (b) single-step in NumPy online (x has
shape (F,)).  `autograd.numpy` behaves as plain NumPy when not being traced.
"""
try:
    import autograd.numpy as anp
except ImportError:  # inference-only fallback for environments without autograd
    import numpy as anp
import numpy as np
from . import sim

K = sim.K_MAX
POS_SCALE = 5.0


# ----------------------------------------------------------------------------
# Featurization
# ----------------------------------------------------------------------------
def feat_dim(use_mask=True, use_cov=False):
    d = 2 + 4 * K + (1 if use_mask else 0) + 4
    if use_cov:
        d += 3
    return d


def featurize_step(u_odo, z, mask, anchor, dt, use_mask=True, Pdiag=None):
    """One feature row.  `anchor` = base-observer pose (DR or EKF) at this step."""
    parts = [np.array([u_odo[0] * dt, u_odo[1] * dt])]
    lm = []
    sr = 6.0
    for i in range(K):
        m = mask[i]
        lm += [(z[i, 0] / sr) * m, np.sin(z[i, 1]) * m, np.cos(z[i, 1]) * m, m]
    parts.append(np.array(lm))
    if use_mask:
        parts.append(np.array([float(mask.max() > 0.5)]))
    ax, ay, ath = anchor
    parts.append(np.array([ax / POS_SCALE, ay / POS_SCALE, np.sin(ath), np.cos(ath)]))
    if Pdiag is not None:
        parts.append(np.array(Pdiag))
    return np.concatenate(parts)


# ----------------------------------------------------------------------------
# Parameter init
# ----------------------------------------------------------------------------
def _rand(shape, scale, rng):
    return (rng.standard_normal(shape) * scale).astype(np.float64)


def init_gru(F, d, rng):
    s = 1.0 / np.sqrt(d + F)
    p = {}
    for g in ("Wz", "Wr", "Wh"):
        p[g] = _rand((d, d + F), s, rng)
        p["b" + g[1]] = np.zeros(d)
    p["Wo"] = _rand((3, d), 1e-3, rng)      # tiny readout -> starts ~= anchor
    p["bo"] = np.zeros(3)
    return p


def init_ssm(F, d, rng, selective=True):
    s = 1.0 / np.sqrt(F)
    p = {}
    p["A_log"] = np.log(np.exp(np.linspace(0.2, 1.5, d)) - 1.0)  # softplus^-1
    p["Win"] = _rand((d, F), s, rng)
    p["Wdelta"] = _rand((d, F), s if selective else 0.0, rng)
    p["bdelta"] = np.zeros(d) - 1.0
    p["WB"] = _rand((d, F), s if selective else 0.0, rng)
    p["bB"] = np.ones(d)
    p["WC"] = _rand((d, F), s if selective else 0.0, rng)
    p["bC"] = np.ones(d)
    p["Wo"] = _rand((3, d), 1e-3, rng)      # tiny readout -> starts ~= anchor
    p["bo"] = np.zeros(3)
    return p


# ----------------------------------------------------------------------------
# Cells  (x: (...,F), h: (...,d))
# ----------------------------------------------------------------------------
def _lin(x, W, b):
    return anp.dot(x, W.T) + b


def _sigmoid(x):
    return 1.0 / (1.0 + anp.exp(-x))


def _softplus(x):
    return anp.log1p(anp.exp(-anp.abs(x))) + anp.maximum(x, 0.0)


def gru_cell(p, x, h):
    hx = anp.concatenate([h, x], axis=-1)
    z = _sigmoid(_lin(hx, p["Wz"], p["bz"]))
    r = _sigmoid(_lin(hx, p["Wr"], p["br"]))
    hxr = anp.concatenate([r * h, x], axis=-1)
    hh = anp.tanh(_lin(hxr, p["Wh"], p["bh"]))
    h_new = (1 - z) * h + z * hh
    y = _lin(h_new, p["Wo"], p["bo"])
    return h_new, y


def ssm_cell(p, x, h, selective=True):
    xin = anp.dot(x, p["Win"].T)
    if selective:
        delta = _softplus(_lin(x, p["Wdelta"], p["bdelta"]))
        B = _lin(x, p["WB"], p["bB"])
        C = _lin(x, p["WC"], p["bC"])
    else:
        delta = _softplus(p["bdelta"])
        B = p["bB"]
        C = p["bC"]
    A = -_softplus(p["A_log"])
    Abar = anp.exp(delta * A)          # (0,1): ~1 retain, small forget
    h_new = Abar * h + (delta * B) * xin
    y = _lin(C * h_new, p["Wo"], p["bo"])
    return h_new, y


# ----------------------------------------------------------------------------
# Adam
# ----------------------------------------------------------------------------
class Adam:
    def __init__(self, keys, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: 0.0 for k in keys}
        self.v = {k: 0.0 for k in keys}
        self.t = 0

    def step(self, params, grads, clip=5.0):
        self.t += 1
        # global grad-norm clip
        tot = np.sqrt(sum(float(np.sum(np.asarray(g) ** 2)) for g in grads.values()) + 1e-12)
        scale = min(1.0, clip / tot)
        for k in self.m:
            g = np.asarray(grads[k]) * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] - self.lr * mh / (np.sqrt(vh) + self.eps)
        return params
