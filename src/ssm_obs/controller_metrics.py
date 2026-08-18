"""Controller-aware offline metrics for selecting state observers.

Pose RMSE weights every state direction uniformly. A feedback controller does not:
it is sensitive only to error directions that change the commanded action.  This
module provides two log-only metrics that can be computed without rolling the
candidate observer inside the plant:

1. Controller Disagreement (CD): exact normalized command discrepancy between
   actions computed from the estimate and from ground truth.
2. Local Control Sensitivity Error (LCSE): a first-order approximation based on
   the controller Jacobian at the true state.

Both use a time-indexed controller to avoid giving candidates different target
progress during offline replay.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping
import numpy as np
from . import dynamics as dyn

Array = np.ndarray
Controller = Callable[[Array, int], Array]


@dataclass(frozen=True)
class ControlScale:
    """Normalization for heterogeneous action units."""
    v: float = 2.5
    omega: float = 3.0

    @property
    def vector(self) -> Array:
        return np.array([max(self.v, 1e-12), max(self.omega, 1e-12)], dtype=float)


def state_error(S: Array, E: Array) -> Array:
    """Return [dx, dy, wrapped dtheta] with shape (T, 3)."""
    S = np.asarray(S, dtype=float)
    E = np.asarray(E, dtype=float)
    if S.shape != E.shape or S.ndim != 2 or S.shape[1] != 3:
        raise ValueError(f"Expected matching (T,3) arrays, got {S.shape} and {E.shape}")
    err = E - S
    err[:, 2] = dyn.wrap(err[:, 2])
    return err


def controller_actions(controller: Controller, poses: Array) -> Array:
    """Evaluate a time-indexed controller on a pose sequence."""
    poses = np.asarray(poses, dtype=float)
    actions = [np.asarray(controller(p, k), dtype=float) for k, p in enumerate(poses)]
    out = np.stack(actions)
    if out.shape != (len(poses), 2):
        raise ValueError(f"Controller must return (2,), got {out.shape}")
    return out


def controller_disagreement(
    S: Array,
    E: Array,
    controller_factory: Callable[[], Controller],
    scale: ControlScale = ControlScale(),
) -> float:
    """RMS normalized action discrepancy on a logged trajectory.

    CD = sqrt(mean_t ||D^{-1}(pi(E_t,t)-pi(S_t,t))||_2^2),
    where D contains actuator ranges. Lower is better.
    """
    c_true = controller_factory()
    c_est = controller_factory()
    u_true = controller_actions(c_true, S)
    u_est = controller_actions(c_est, E)
    du = (u_est - u_true) / scale.vector
    return float(np.sqrt(np.mean(np.sum(du * du, axis=1))))


def finite_difference_jacobian(
    controller: Controller,
    pose: Array,
    k: int,
    eps: Array | None = None,
) -> Array:
    """Central-difference Jacobian d pi / d [x,y,theta]."""
    pose = np.asarray(pose, dtype=float)
    eps = np.asarray(eps if eps is not None else [1e-3, 1e-3, 1e-4], dtype=float)
    J = np.zeros((2, 3), dtype=float)
    for j in range(3):
        plus = pose.copy(); plus[j] += eps[j]
        minus = pose.copy(); minus[j] -= eps[j]
        plus[2] = dyn.wrap(plus[2]); minus[2] = dyn.wrap(minus[2])
        J[:, j] = (np.asarray(controller(plus, k)) - np.asarray(controller(minus, k))) / (2 * eps[j])
    return J


def local_control_sensitivity_error(
    S: Array,
    E: Array,
    controller_factory: Callable[[], Controller],
    scale: ControlScale = ControlScale(),
) -> float:
    """First-order controller-weighted state error.

    LCSE = sqrt(mean_t ||D^{-1} J_pi(S_t,t) (E_t-S_t)||_2^2).
    Lower is better.  Unlike exact CD, LCSE exposes the local sensitivity
    interpretation and can be precomputed as a task metric tensor J^T J.
    """
    err = state_error(S, E)
    controller = controller_factory()
    vals = []
    for k, (s, e) in enumerate(zip(S, err)):
        J = finite_difference_jacobian(controller, s, k)
        vals.append((J @ e) / scale.vector)
    vals = np.asarray(vals)
    return float(np.sqrt(np.mean(np.sum(vals * vals, axis=1))))


def proxy_metrics(
    S: Array,
    E: Array,
    controller_factory: Callable[[], Controller],
    scale: ControlScale = ControlScale(),
) -> Dict[str, float]:
    """Compute all offline observer-selection proxies."""
    from .metrics import pos_rmse, head_rmse
    return {
        "pose_rmse": pos_rmse(S, E),
        "heading_rmse": head_rmse(S, E),
        "controller_disagreement": controller_disagreement(S, E, controller_factory, scale),
        "lcse": local_control_sensitivity_error(S, E, controller_factory, scale),
    }


def selection_summary(
    conditions: Iterable[str],
    observers: Iterable[str],
    proxy: Mapping[str, Mapping[str, float]],
    closed_loop: Mapping[str, Mapping[str, float]],
) -> Dict[str, object]:
    """Top-1 mismatch and closed-loop regret induced by an offline proxy."""
    observers = list(observers)
    rows = []
    for condition in conditions:
        chosen = min(observers, key=lambda n: proxy[condition][n])
        oracle = min(observers, key=lambda n: closed_loop[condition][n])
        chosen_value = float(closed_loop[condition][chosen])
        oracle_value = float(closed_loop[condition][oracle])
        rows.append({
            "condition": condition,
            "chosen": chosen,
            "oracle": oracle,
            "flip": chosen != oracle,
            "regret": chosen_value - oracle_value,
            "relative_regret": (chosen_value - oracle_value) / max(oracle_value, 1e-12),
        })
    regrets = np.array([r["regret"] for r in rows], dtype=float)
    rel = np.array([r["relative_regret"] for r in rows], dtype=float)
    return {
        "n_conditions": len(rows),
        "flip_fraction": float(np.mean([r["flip"] for r in rows])),
        "mean_regret": float(regrets.mean()),
        "median_regret": float(np.median(regrets)),
        "max_regret": float(regrets.max()),
        "mean_relative_regret": float(rel.mean()),
        "max_relative_regret": float(rel.max()),
        "rows": rows,
    }


def counterfactual_error_replay(
    S: Array,
    E: Array,
    path: Array,
    controller_factory: Callable[[], Controller],
    dt: float,
    control_low: Array | None = None,
    control_high: Array | None = None,
) -> float:
    """Offline counterfactual rollout using the candidate's logged error trace.

    Let e_t = E_t minus S_t be the observer error measured on a common replay.
    A shadow plant starts from the logged initial state. At each step the
    controller receives shadow_state plus e_t, and the nominal dynamics advance
    under the resulting command. The returned cross-track RMSE estimates how
    the *temporal structure* of errors would affect feedback, without rerunning
    sensing or the observer in closed loop.

    This is more informative than instantaneous command disagreement because it
    accumulates heading-induced drift and exposes oscillatory or destabilizing
    error sequences. It requires only a controller and nominal dynamics.
    """
    from .metrics import cross_track_rmse
    S = np.asarray(S, dtype=float); path = np.asarray(path, dtype=float)
    err = state_error(S, E)
    low = np.asarray(control_low if control_low is not None else [0.0, -3.0], dtype=float)
    high = np.asarray(control_high if control_high is not None else [2.5, 3.0], dtype=float)
    controller = controller_factory()
    shadow = S[0].copy(); states = np.zeros_like(S)
    for k, e in enumerate(err):
        est_shadow = shadow + e
        est_shadow[2] = dyn.wrap(est_shadow[2])
        u = np.clip(np.asarray(controller(est_shadow, k), dtype=float), low, high)
        states[k] = shadow
        shadow = dyn.step_true(shadow, u, dt)
    return cross_track_rmse(states, path)
