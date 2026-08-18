"""Estimation and control metrics.

Open-loop  : how well the estimate matches the true state (replay).
Closed-loop: how well the ROBOT tracks the path when the controller acts on
             the estimate.  These can disagree -- that disagreement is the
             paper's central result (objective mismatch for observers).
"""
import numpy as np
from . import dynamics as dyn


# ---- estimation quality -----------------------------------------------------
def pos_rmse(S, E):
    return float(np.sqrt(np.mean(np.sum((S[:, :2] - E[:, :2]) ** 2, axis=1))))


def head_rmse(S, E):
    return float(np.sqrt(np.mean(dyn.wrap(S[:, 2] - E[:, 2]) ** 2)))


# ---- closed-loop tracking ---------------------------------------------------
def _cross_track(S, P):
    """Distance from each true position to the nearest point on path P."""
    d = np.linalg.norm(S[:, None, :2] - P[None, :, :], axis=2)
    return d.min(axis=1)


def cross_track_rmse(S, P):
    return float(np.sqrt(np.mean(_cross_track(S, P) ** 2)))


def cross_track_max(S, P):
    return float(_cross_track(S, P).max())


def control_effort(U):
    """Mean squared steering rate -- 'how hard did the controller work'."""
    return float(np.mean(U[:, 1] ** 2))


def recovery_time(S, P, on, dt, tube=0.25):
    """Mean time (s) after a blackout ends for cross-track error to fall back
    inside `tube`.  Returns np.nan if there are no blackouts."""
    ct = _cross_track(S, P)
    n = len(on)
    ends = [k for k in range(1, n) if on[k] and not on[k - 1]]
    if not ends:
        return np.nan
    times = []
    for e in ends:
        k = e
        while k < n and ct[k] > tube:
            k += 1
        times.append((k - e) * dt if k < n else (n - e) * dt)
    return float(np.mean(times))


def divergence(S, P, thresh=1.0):
    """Fraction of steps with cross-track error above `thresh` (lost tracking)."""
    return float(np.mean(_cross_track(S, P) > thresh))


def all_metrics(out, dt, tube=0.25, div_thresh=1.0):
    S, E, P, U, on = out["S"], out["E"], out["P"], out["U"], out["on"]
    return dict(
        pos_rmse=pos_rmse(S, E),
        head_rmse=head_rmse(S, E),
        ct_rmse=cross_track_rmse(S, P),
        ct_max=cross_track_max(S, P),
        effort=control_effort(U),
        recovery=recovery_time(S, P, on, dt, tube),
        diverge=divergence(S, P, div_thresh),
    )
