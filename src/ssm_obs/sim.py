"""Scenario configuration, reference paths, dropout schedules, and rollout.

A "condition" is a SensingParams dataclass that pins the four degradation
axes studied in the paper:
  - landmark density   (n_landmarks in the map)
  - dropout duration   (blackout window length)
  - measurement noise  (sig_r, sig_b)
  - odometry mismatch  (slip, gyro_bias)
"""
from dataclasses import dataclass, field, replace
import numpy as np
from . import dynamics as dyn


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class SensingParams:
    dt: float = 0.1
    # odometry
    sig_v: float = 0.03          # linear-vel noise std (m/s)
    sig_w: float = 0.03          # angular-vel noise std (rad/s)
    slip: float = 1.0            # multiplicative wheel-slip on v (1.0 = none)
    gyro_bias: float = 0.0       # additive angular-rate bias (rad/s)
    # range-bearing sensor
    sensor_range: float = 6.0    # max sensing radius (m)
    sig_r: float = 0.10          # range noise std (m)
    sig_b: float = 0.05          # bearing noise std (rad)
    # intermittent sensing
    dropout_period: int = 40     # steps between the start of blackout windows
    dropout_len: int = 12        # blackout window length (steps)
    # map
    n_landmarks: int = 6         # landmarks placed around the workspace


NOMINAL = SensingParams()


# ----------------------------------------------------------------------------
# Landmark maps  (fixed & known; the observers all see the same map)
# ----------------------------------------------------------------------------
def make_landmarks(n, rng, extent=5.0):
    """Place n landmarks on a ring around the workspace (deterministic-ish)."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ang = ang + rng.normal(0, 0.15, size=n)
    rad = extent + rng.normal(0, 0.4, size=n)
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)


K_MAX = 8
# One fixed, known landmark map used by EVERY observer and condition.
FIXED_MAP = make_landmarks(K_MAX, np.random.default_rng(12345))


def active_mask(n_active, rng=None, randomize=False):
    """Boolean (K_MAX,) subset of landmarks that are 'present' (density axis)."""
    a = np.zeros(K_MAX, dtype=bool)
    idx = np.arange(K_MAX)
    if randomize and rng is not None:
        idx = rng.permutation(K_MAX)
    a[idx[:n_active]] = True
    return a


# ----------------------------------------------------------------------------
# Reference trajectory: figure-eight (lemniscate of Gerono)
# ----------------------------------------------------------------------------
def figure_eight(n_steps, dt, A=3.0, laps=1.0):
    """Return reference path P (n_steps,2) plus ref heading, v_ref, w_ref.

    Curvature reverses twice per lap -> stresses heading estimation.
    """
    T = n_steps * dt
    om = 2 * np.pi * laps / T
    t = np.arange(n_steps) * dt
    x = A * np.sin(om * t)
    y = (A / 2.0) * np.sin(2 * om * t)
    dx = A * om * np.cos(om * t)
    dy = A * om * np.cos(2 * om * t)
    ddx = -A * om * om * np.sin(om * t)
    ddy = -2 * A * om * om * np.sin(2 * om * t)
    th = np.arctan2(dy, dx)
    v = np.hypot(dx, dy)
    w = (dx * ddy - dy * ddx) / (dx * dx + dy * dy + 1e-9)   # v * curvature
    P = np.stack([x, y], axis=1)
    return P, dyn.wrap(th), v, w


# ----------------------------------------------------------------------------
# Dropout schedule
# ----------------------------------------------------------------------------
def dropout_schedule(n_steps, params, rng, jitter=True):
    """Boolean array: True = sensing ON, False = blackout."""
    on = np.ones(n_steps, dtype=bool)
    period = max(params.dropout_period, params.dropout_len + 1)
    start = rng.integers(0, period) if jitter else 0
    s = start
    while s < n_steps:
        e = min(s + params.dropout_len, n_steps)
        on[s:e] = False
        s += period
    return on


# ----------------------------------------------------------------------------
# Rollout with a *given* controller and *given* estimator
# ----------------------------------------------------------------------------
def rollout(controller, estimator, params, landmarks, ref, rng,
            start=None, log=True, active=None):
    """Closed-loop rollout.

    controller(est_pose, k) -> u_cmd           (acts on the ESTIMATE)
    estimator: object with .reset(s0) and .step(u_odo, z, mask, k) -> est_pose

    Returns a dict of logged arrays.
    """
    P, th_ref, v_ref, w_ref = ref
    n = len(P)
    dt = params.dt
    on = dropout_schedule(n, params, rng)

    if start is None:
        s = np.array([P[0, 0], P[0, 1], th_ref[0]])
    else:
        s = np.array(start, float)
    estimator.reset(s.copy())

    S = np.zeros((n, 3)); E = np.zeros((n, 3)); U = np.zeros((n, 2))
    Uodo = np.zeros((n, 2)); Z = np.zeros((n, len(landmarks), 2))
    M = np.zeros((n, len(landmarks)))

    est_pose = s.copy()
    for k in range(n):
        u = controller(est_pose, k)                 # controller uses ESTIMATE
        u = np.clip(u, [-0.0, -3.0], [2.5, 3.0])     # actuator limits
        u_odo = dyn.odometry(u, params, rng)
        z, mask = dyn.sense(s, landmarks, params, on[k], rng, active)
        est_pose = estimator.step(u_odo, z, mask, k)
        S[k] = s; E[k] = est_pose; U[k] = u
        Uodo[k] = u_odo; Z[k] = z; M[k] = mask
        s = dyn.step_true(s, u, dt)                  # TRUE dynamics advance

    out = dict(S=S, E=E, U=U, Uodo=Uodo, Z=Z, M=M, on=on,
               P=P, th_ref=th_ref, v_ref=v_ref, w_ref=w_ref,
               landmarks=landmarks)
    return out


def rollout_openloop(estimator, params, landmarks, control_seq, ref, rng,
                     start=None, active=None):
    """Open-loop replay: the controls are FIXED (from `control_seq`), the
    estimator just estimates.  Used for open-loop RMSE and for logging
    training data.  Returns the same dict shape as rollout()."""
    P, th_ref, v_ref, w_ref = ref
    n = len(control_seq)
    dt = params.dt
    on = dropout_schedule(n, params, rng)
    s = np.array([P[0, 0], P[0, 1], th_ref[0]]) if start is None else np.array(start, float)
    estimator.reset(s.copy())

    S = np.zeros((n, 3)); E = np.zeros((n, 3)); U = np.zeros((n, 2))
    Uodo = np.zeros((n, 2)); Z = np.zeros((n, len(landmarks), 2))
    M = np.zeros((n, len(landmarks)))
    for k in range(n):
        u = control_seq[k]
        u_odo = dyn.odometry(u, params, rng)
        z, mask = dyn.sense(s, landmarks, params, on[k], rng, active)
        est_pose = estimator.step(u_odo, z, mask, k)
        S[k] = s; E[k] = est_pose; U[k] = u
        Uodo[k] = u_odo; Z[k] = z; M[k] = mask
        s = dyn.step_true(s, u, dt)
    return dict(S=S, E=E, U=U, Uodo=Uodo, Z=Z, M=M, on=on,
                P=P, th_ref=th_ref, v_ref=v_ref, w_ref=w_ref,
                landmarks=landmarks)
