"""Classical observers: dead reckoning and the EKF.

Observer interface used everywhere:
    obs.reset(s0)                         # s0 = true start pose (3,)
    est = obs.step(u_odo, z, mask, k)     # returns estimated pose (3,)
"""
import numpy as np
from . import dynamics as dyn


class DeadReckoning:
    """Integrate odometry from the known start pose.  Lower-bound baseline."""
    def __init__(self, params):
        self.dt = params.dt

    def reset(self, s0):
        self.s = s0.copy()

    def step(self, u_odo, z, mask, k):
        self.s = dyn.step_true(self.s, u_odo, self.dt)
        return self.s.copy()


class EKF:
    """Extended Kalman filter with a unicycle motion model and range-bearing
    updates to a known landmark map.

    The filter is given *assumed* process/measurement covariances (Q, R).
    Under odometry bias / wheel slip / long dropout these assumptions are
    violated -> the analytic baseline that learned observers must beat where
    it matters.
    """
    def __init__(self, params, landmarks, q_scale=1.0, r_scale=1.0):
        self.dt = params.dt
        self.L = landmarks
        self.p = params
        # assumed process noise (on x,y,theta) -- derived from odom noise
        qv = (params.sig_v * params.dt) ** 2
        qw = (params.sig_w * params.dt) ** 2
        self.Q = q_scale * np.diag([qv, qv, qw + 1e-6])
        self.R = r_scale * np.diag([params.sig_r ** 2, params.sig_b ** 2])

    def reset(self, s0):
        self.mu = s0.copy()
        self.P = np.diag([0.05, 0.05, 0.02])

    def _predict(self, u_odo):
        v, w = u_odo
        th = self.mu[2]
        dt = self.dt
        self.mu = dyn.step_true(self.mu, u_odo, dt)
        F = np.array([[1, 0, -v * np.sin(th) * dt],
                      [0, 1,  v * np.cos(th) * dt],
                      [0, 0, 1]])
        self.P = F @ self.P @ F.T + self.Q

    def _update(self, z, mask):
        for i in range(len(self.L)):
            if mask[i] < 0.5:
                continue
            zhat = dyn.meas_predict(self.mu, self.L[i])
            H = dyn.meas_jacobian(self.mu, self.L[i])
            y = z[i] - zhat
            y[1] = dyn.wrap(y[1])
            S = H @ self.P @ H.T + self.R
            K = self.P @ H.T @ np.linalg.inv(S)
            self.mu = self.mu + K @ y
            self.mu[2] = dyn.wrap(self.mu[2])
            self.P = (np.eye(3) - K @ H) @ self.P

    def step(self, u_odo, z, mask, k):
        self._predict(u_odo)
        if mask.sum() > 0:
            self._update(z, mask)
        return self.mu.copy()
