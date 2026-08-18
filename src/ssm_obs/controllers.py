"""Geometric tracking controllers used in closed-loop and offline evaluation."""
from __future__ import annotations
import numpy as np
from . import dynamics as dyn


class PurePursuit:
    """Stateful path-progress pure pursuit for the actual closed loop."""
    def __init__(self, P, v_cmd=1.0, Ld=0.8):
        self.P = np.asarray(P, dtype=float)
        self.v = float(v_cmd)
        self.Ld = float(Ld)
        self._i = 0

    def reset(self):
        self._i = 0

    def __call__(self, pose, k):
        x, y, th = pose
        i = self._i
        N = len(self.P)
        while i < N - 1 and np.hypot(*(self.P[i] - [x, y])) < self.Ld:
            i += 1
        self._i = i
        gx, gy = self.P[i]
        alpha = dyn.wrap(np.arctan2(gy - y, gx - x) - th)
        kappa = 2.0 * np.sin(alpha) / max(self.Ld, 1e-9)
        return np.array([self.v, self.v * kappa])


class TimeIndexedPurePursuit:
    """Stateless pure-pursuit policy for comparable logged-data metrics.

    The target point is selected from reference progress k rather than from the
    candidate estimate. Thus all observers are evaluated against the same task
    state and differences reflect control-relevant pose error, not target drift.
    """
    def __init__(self, P, v_cmd=1.0, Ld=0.8):
        self.P = np.asarray(P, dtype=float)
        self.v = float(v_cmd)
        self.Ld = float(Ld)
        seg = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.arc = np.concatenate([[0.0], np.cumsum(seg)])

    def __call__(self, pose, k):
        k = min(max(int(k), 0), len(self.P) - 1)
        target_arc = self.arc[k] + self.Ld
        i = int(np.searchsorted(self.arc, target_arc, side="left"))
        i = min(i, len(self.P) - 1)
        x, y, th = pose
        gx, gy = self.P[i]
        alpha = dyn.wrap(np.arctan2(gy - y, gx - x) - th)
        return np.array([self.v, self.v * 2.0 * np.sin(alpha) / max(self.Ld, 1e-9)])


class Kanayama:
    """Kanayama et al. (ICRA 1990) time-indexed posture controller."""
    def __init__(self, ref, Kx=1.5, Ky=6.0, Kth=3.0):
        self.P, self.th_ref, self.v_ref, self.w_ref = [np.asarray(x) for x in ref]
        self.Kx, self.Ky, self.Kth = Kx, Ky, Kth

    def __call__(self, pose, k):
        k = min(max(int(k), 0), len(self.P) - 1)
        x, y, th = pose
        xr, yr = self.P[k]; thr = self.th_ref[k]
        c, s = np.cos(th), np.sin(th)
        ex = c * (xr - x) + s * (yr - y)
        ey = -s * (xr - x) + c * (yr - y)
        eth = dyn.wrap(thr - th)
        v = self.v_ref[k] * np.cos(eth) + self.Kx * ex
        w = self.w_ref[k] + self.v_ref[k] * (self.Ky * ey + self.Kth * np.sin(eth))
        return np.array([v, w])
