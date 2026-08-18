"""Differential-drive (unicycle) dynamics, odometry, and range-bearing sensing.

State s = [x, y, theta].  Control u = [v, omega].
All angles are wrapped to (-pi, pi].  Everything here is plain NumPy.
"""
import numpy as np


def wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def step_true(s, u, dt):
    """One exact unicycle step (no noise).  s=[x,y,theta], u=[v,omega]."""
    x, y, th = s
    v, w = u
    return np.array([x + v * np.cos(th) * dt,
                     y + v * np.sin(th) * dt,
                     wrap(th + w * dt)])


def odometry(u_true, params, rng):
    """Corrupt the true control into a measured odometry reading.

    v_meas = slip * v_true + N(0, sig_v^2)
    w_meas =        w_true + gyro_bias + N(0, sig_w^2)

    `slip` (<1 = wheel slip) and `gyro_bias` are the *systematic* model
    mismatch the EKF is not told about.
    """
    v, w = u_true
    v_m = params.slip * v + rng.normal(0.0, params.sig_v)
    w_m = w + params.gyro_bias + rng.normal(0.0, params.sig_w)
    return np.array([v_m, w_m])


def sense(s, landmarks, params, sensing_on, rng, active=None):
    """Range-bearing measurements to a *fixed, known* landmark map.

    `active` is an optional boolean (K,) subset mask (the "density" axis):
    inactive landmarks are never observed.  This keeps the map size (K) --
    and hence the learned observer's input width -- fixed across conditions.

    Returns (z, mask):
      z    : (K, 2) array of [range, bearing] (bearing relative to heading)
      mask : (K,)  1.0 if landmark observed this step, else 0.0
    A landmark is observed iff active AND sensing_on AND within sensor_range.
    """
    x, y, th = s
    K = len(landmarks)
    z = np.zeros((K, 2))
    mask = np.zeros(K)
    if not sensing_on:
        return z, mask
    for i, (lx, ly) in enumerate(landmarks):
        if active is not None and not active[i]:
            continue
        dx, dy = lx - x, ly - y
        rng_true = np.hypot(dx, dy)
        if rng_true > params.sensor_range:
            continue
        r = rng_true + rng.normal(0.0, params.sig_r)
        b = wrap(np.arctan2(dy, dx) - th) + rng.normal(0.0, params.sig_b)
        z[i] = [r, wrap(b)]
        mask[i] = 1.0
    return z, mask


def meas_predict(s, lm):
    """Noise-free predicted range-bearing to one landmark lm=(lx,ly)."""
    x, y, th = s
    dx, dy = lm[0] - x, lm[1] - y
    r = np.hypot(dx, dy)
    b = wrap(np.arctan2(dy, dx) - th)
    return np.array([r, b])


def meas_jacobian(s, lm):
    """Jacobian d[r,b]/d[x,y,theta] for one landmark (for the EKF)."""
    x, y, th = s
    dx, dy = lm[0] - x, lm[1] - y
    q = dx * dx + dy * dy
    r = np.sqrt(q)
    return np.array([[-dx / r,     -dy / r,      0.0],
                     [ dy / q,     -dx / q,     -1.0]])
