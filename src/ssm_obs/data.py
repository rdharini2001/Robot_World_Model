"""Training-data generation.

Trajectories are produced by an EXPERT rollout: pure-pursuit closes the loop
on the TRUE pose, so the visited-state distribution matches closed-loop
operation.  Each trajectory samples a random 'condition' (the four
degradation axes) so the learned observers generalize across the eval sweeps
rather than overfitting one setting.
"""
import numpy as np
from . import sim, dynamics as dyn, controllers, ekf as ekf_mod, nn_core as nc


def sample_condition(rng):
    return sim.SensingParams(
        sig_v=rng.uniform(0.02, 0.05),
        sig_w=rng.uniform(0.02, 0.05),
        slip=rng.uniform(0.82, 1.0),
        gyro_bias=rng.uniform(-0.15, 0.15),
        sig_r=rng.uniform(0.05, 0.35),
        sig_b=rng.uniform(0.03, 0.18),
        dropout_len=int(rng.integers(6, 28)),
        dropout_period=int(rng.integers(30, 70)),
        n_landmarks=int(rng.integers(2, sim.K_MAX + 1)),
    )


def expert_rollout(params, n_steps, rng):
    """Pure-pursuit on the TRUE pose; log odometry, measurements, masks."""
    A = rng.uniform(2.5, 3.5)
    ref = sim.figure_eight(n_steps, params.dt, A=A, laps=1.0)
    P, th_ref, v_ref, w_ref = ref
    active = sim.active_mask(params.n_landmarks, rng, randomize=True)
    on = sim.dropout_schedule(n_steps, params, rng)
    ctrl = controllers.PurePursuit(P, v_cmd=rng.uniform(0.9, 1.2), Ld=0.8)
    lm = sim.FIXED_MAP

    s = np.array([P[0, 0], P[0, 1], th_ref[0]])
    S = np.zeros((n_steps, 3)); Uodo = np.zeros((n_steps, 2))
    Z = np.zeros((n_steps, sim.K_MAX, 2)); M = np.zeros((n_steps, sim.K_MAX))
    for k in range(n_steps):
        u = ctrl(s, k)
        u = np.clip(u, [0.0, -3.0], [2.5, 3.0])
        u_odo = dyn.odometry(u, params, rng)
        z, mask = dyn.sense(s, lm, params, on[k], rng, active)
        S[k] = s; Uodo[k] = u_odo; Z[k] = z; M[k] = mask
        s = dyn.step_true(s, u, params.dt)
    return dict(S=S, Uodo=Uodo, Z=Z, M=M, s0=S[0].copy(), params=params)


def generate(n_traj, n_steps, seed):
    rng = np.random.default_rng(seed)
    return [expert_rollout(sample_condition(rng), n_steps, rng) for _ in range(n_traj)]


# ---- imagination / long-horizon-blackout curriculum augmentation -----------
def augment_long_blackout(traj, rng, min_len=32, max_len=58):
    """Return a COPY of `traj` with one contiguous synthetic sensor blackout
    of length in [min_len, max_len] carved into it (Z, M zeroed over that
    span). S and Uodo are untouched: in this simulator, sensing never
    influences the true dynamics or the odometry stream, so this is a valid
    counterfactual "what if the map had gone dark for longer here" example.

    Used to build a training curriculum for the imagination-augmented
    ("world-model") observer arm: it forces the network to see -- and be
    scored through -- gaps considerably longer than the naturally sampled
    dropout_len in `sample_condition` (max 27), so it is explicitly trained
    for the extended blind-imagination regime the dropout_len sweep probes
    at evaluation time (up to 34 steps).
    """
    T = len(traj["S"])
    H = int(rng.integers(min_len, max_len + 1))
    H = min(H, T - 2)
    start = int(rng.integers(1, max(2, T - H - 1)))
    out = dict(traj)
    Z = traj["Z"].copy(); M = traj["M"].copy()
    Z[start:start + H] = 0.0
    M[start:start + H] = 0.0
    out["Z"] = Z; out["M"] = M
    return out


def generate_imagination_curriculum(n_traj, n_steps, seed, augment_frac=0.5,
                                     min_len=32, max_len=58):
    """Training set for the imagination-augmented observer arm: the same
    randomized-condition trajectories as `generate`, with a fraction of them
    additionally carrying one synthetic long blackout window."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_traj):
        traj = expert_rollout(sample_condition(rng), n_steps, rng)
        if rng.random() < augment_frac:
            traj = augment_long_blackout(traj, rng, min_len, max_len)
        out.append(traj)
    return out


# ---- turn logged trajectories into (X, anchors, targets) --------------------
def _dr_anchors(traj):
    dr = ekf_mod.DeadReckoning(traj["params"])
    dr.reset(traj["s0"])
    return np.stack([dr.step(traj["Uodo"][k], traj["Z"][k], traj["M"][k], k)
                     for k in range(len(traj["S"]))])


def _ekf_anchors(traj):
    e = ekf_mod.EKF(traj["params"], sim.FIXED_MAP)
    e.reset(traj["s0"])
    anch = np.zeros((len(traj["S"]), 3)); Pd = np.zeros((len(traj["S"]), 3))
    for k in range(len(traj["S"])):
        a = e.step(traj["Uodo"][k], traj["Z"][k], traj["M"][k], k)
        anch[k] = a; Pd[k] = np.diag(e.P)
    return anch, Pd


def featurize(dataset, anchor="dr", use_mask=True, use_cov=False):
    X, A, Y = [], [], []
    for traj in dataset:
        T = len(traj["S"])
        if anchor == "ekf":
            anch, Pd = _ekf_anchors(traj)
        else:
            anch = _dr_anchors(traj); Pd = None
        rows = []
        for k in range(T):
            pdiag = Pd[k] if (use_cov and Pd is not None) else None
            rows.append(nc.featurize_step(traj["Uodo"][k], traj["Z"][k],
                                          traj["M"][k], anch[k], traj["params"].dt,
                                          use_mask=use_mask, Pdiag=pdiag))
        X.append(np.stack(rows)); A.append(anch); Y.append(traj["S"])
    return np.array(X), np.array(A), np.array(Y)
