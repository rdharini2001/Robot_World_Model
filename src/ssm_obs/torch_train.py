"""PyTorch training backend for the recurrent residual observers.

The original repository used HIPS autograd. That dependency is unmaintained on
some modern Python/NumPy stacks, so this backend reproduces the same cells and
exports the exact same NumPy checkpoint format used by online inference.
"""
from __future__ import annotations
import copy
import time
from typing import Dict
import numpy as np
import torch
from torch import nn
from . import data as data_mod


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x))


class ResidualSequenceModel(nn.Module):
    def __init__(self, F: int, d: int, cell: str, selective: bool, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.F, self.d, self.cell, self.selective = F, d, cell, selective
        if cell == "gru":
            s = 1.0 / np.sqrt(d + F)
            for name in ("Wz", "Wr", "Wh"):
                self.register_parameter(name, nn.Parameter(torch.randn(d, d + F, dtype=torch.float64) * s))
                self.register_parameter("b" + name[1], nn.Parameter(torch.zeros(d, dtype=torch.float64)))
        elif cell == "ssm":
            s = 1.0 / np.sqrt(F)
            init_A = _inv_softplus(torch.linspace(0.2, 1.5, d, dtype=torch.float64))
            self.A_log = nn.Parameter(init_A)
            self.Win = nn.Parameter(torch.randn(d, F, dtype=torch.float64) * s)
            gate_scale = s if selective else 0.0
            self.Wdelta = nn.Parameter(torch.randn(d, F, dtype=torch.float64) * gate_scale)
            self.bdelta = nn.Parameter(torch.full((d,), -1.0, dtype=torch.float64))
            self.WB = nn.Parameter(torch.randn(d, F, dtype=torch.float64) * gate_scale)
            self.bB = nn.Parameter(torch.ones(d, dtype=torch.float64))
            self.WC = nn.Parameter(torch.randn(d, F, dtype=torch.float64) * gate_scale)
            self.bC = nn.Parameter(torch.ones(d, dtype=torch.float64))
        else:
            raise ValueError(f"Unknown cell {cell}")
        self.Wo = nn.Parameter(torch.randn(3, d, dtype=torch.float64) * 1e-3)
        self.bo = nn.Parameter(torch.zeros(3, dtype=torch.float64))

    @staticmethod
    def _lin(x, W, b):
        return x @ W.T + b

    def step(self, x, h):
        if self.cell == "gru":
            hx = torch.cat([h, x], dim=-1)
            z = torch.sigmoid(self._lin(hx, self.Wz, self.bz))
            r = torch.sigmoid(self._lin(hx, self.Wr, self.br))
            hxr = torch.cat([r * h, x], dim=-1)
            hh = torch.tanh(self._lin(hxr, self.Wh, self.bh))
            h_new = (1.0 - z) * h + z * hh
            y = self._lin(h_new, self.Wo, self.bo)
            return h_new, y

        xin = x @ self.Win.T
        if self.selective:
            delta = torch.nn.functional.softplus(self._lin(x, self.Wdelta, self.bdelta))
            B = self._lin(x, self.WB, self.bB)
            C = self._lin(x, self.WC, self.bC)
        else:
            delta = torch.nn.functional.softplus(self.bdelta)
            B, C = self.bB, self.bC
        A = -torch.nn.functional.softplus(self.A_log)
        Abar = torch.exp(delta * A)
        h_new = Abar * h + (delta * B) * xin
        y = self._lin(C * h_new, self.Wo, self.bo)
        return h_new, y

    def forward(self, X, anchors):
        B, T, _ = X.shape
        h = X.new_zeros((B, self.d))
        outs = []
        for t in range(T):
            h, y = self.step(X[:, t], h)
            outs.append(anchors[:, t] + y)
        return torch.stack(outs, dim=1)

    def export_numpy(self) -> Dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().copy() for k, v in self.named_parameters()}


def _loss(model, X, A, Y, lam):
    E = model(X, A)
    dpos = E[..., :2] - Y[..., :2]
    lpos = torch.mean(torch.sum(dpos * dpos, dim=-1))
    lhead = torch.mean(1.0 - torch.cos(E[..., 2] - Y[..., 2]))
    return lpos + lam * lhead


def train(spec, dataset, val_dataset=None, verbose=True, device="cpu"):
    """Train one ``Spec`` and return the legacy NumPy parameter dictionary."""
    torch.set_num_threads(1)
    X, A, Y = data_mod.featurize(dataset, spec.anchor, spec.use_mask, spec.use_cov)
    X = torch.as_tensor(X, dtype=torch.float64, device=device)
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    Y = torch.as_tensor(Y, dtype=torch.float64, device=device)
    if val_dataset is not None:
        Xv, Av, Yv = data_mod.featurize(val_dataset, spec.anchor, spec.use_mask, spec.use_cov)
        Xv = torch.as_tensor(Xv, dtype=torch.float64, device=device)
        Av = torch.as_tensor(Av, dtype=torch.float64, device=device)
        Yv = torch.as_tensor(Yv, dtype=torch.float64, device=device)
    model = ResidualSequenceModel(X.shape[-1], spec.d, spec.cell, spec.selective, spec.seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=spec.lr)
    rng = np.random.default_rng(spec.seed)
    best_val = float("inf"); best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()
    for ep in range(spec.epochs):
        model.train(); order = rng.permutation(len(X)); tr_sum = 0.0
        for start in range(0, len(X), spec.batch):
            idx = torch.as_tensor(order[start:start + spec.batch], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            loss = _loss(model, X[idx], A[idx], Y[idx], spec.lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_sum += float(loss.detach()) * len(idx)
        tr = tr_sum / len(X)
        model.eval()
        with torch.no_grad():
            vl = float(_loss(model, Xv, Av, Yv, spec.lam)) if val_dataset is not None else tr
        if vl < best_val:
            best_val = vl; best_state = copy.deepcopy(model.state_dict())
        if verbose and (ep % 5 == 0 or ep == spec.epochs - 1):
            print(f"  [{spec.name}] ep{ep:02d} train={tr:.4f} val={vl:.4f} ({time.time()-t0:.0f}s)")
    model.load_state_dict(best_state)
    return model.export_numpy()
