"""Experiment orchestration: the model zoo, an observer factory, evaluation
conditions, closed-loop sweeps, and the objective-mismatch analysis."""
import numpy as np
from . import sim, ekf, controllers, metrics, models, data as data_mod
from .train import Spec

N_STEPS = 220
A_PATH = 3.0

# ---- model zoo --------------------------------------------------------------
ZOO = [
    Spec(name="gru_dr",        cell="gru", anchor="dr",  d=32, epochs=32, lr=4e-3),
    Spec(name="ssm_dr",        cell="ssm", anchor="dr",  d=32, epochs=32, lr=4e-3, selective=True),
    Spec(name="ssm_ekf",       cell="ssm", anchor="ekf", d=32, epochs=32, lr=3e-3, selective=True, use_cov=True),
    Spec(name="gru_ekf",       cell="gru", anchor="ekf", d=32, epochs=32, lr=3e-3, use_cov=True),
    Spec(name="ssm_dr_nosel",  cell="ssm", anchor="dr",  d=32, epochs=32, lr=4e-3, selective=False),
    Spec(name="ssm_dr_nomask", cell="ssm", anchor="dr",  d=32, epochs=32, lr=4e-3, selective=True, use_mask=False),
    Spec(name="ssm_ekf_nosel", cell="ssm", anchor="ekf", d=32, epochs=32, lr=3e-3, selective=False, use_cov=True),
    Spec(name="ssm_ekf_nomask", cell="ssm", anchor="ekf", d=32, epochs=32, lr=3e-3, selective=True, use_mask=False, use_cov=True),
]

# pretty names + the 5 "core" observers shown in the main sweeps
PRETTY = {
    "dead_reckoning": "Dead reckoning",
    "ekf": "EKF",
    "gru_dr": "GRU (DR-residual)",
    "ssm_dr": "Selective SSM (DR-residual)",
    "ssm_ekf": "Selective SSM (EKF-residual)",
    "gru_ekf": "GRU (EKF-residual)",
    "ssm_dr_nosel": "SSM, no selectivity",
    "ssm_dr_nomask": "SSM, no mask (DR anchor)",
    "ssm_ekf_nosel": "SSM-EKF, no selectivity",
    "ssm_ekf_nomask": "SSM-EKF, no mask",
}
CORE = ["dead_reckoning", "ekf", "gru_dr", "ssm_dr", "ssm_ekf", "gru_ekf"]
ABLATIONS = ["ssm_ekf", "ssm_ekf_nosel", "ssm_ekf_nomask"]


# ---- observer factory -------------------------------------------------------
def build(name, trained, params_sim):
    lm = sim.FIXED_MAP
    if name == "dead_reckoning":
        return ekf.DeadReckoning(params_sim)
    if name == "ekf":
        return ekf.EKF(params_sim, lm)
    spec, p = trained[name]
    return models.LearnedObserver(p, spec.cell, spec.anchor, params_sim, lm,
                                  use_mask=spec.use_mask, selective=spec.selective,
                                  use_cov=spec.use_cov)


# ---- evaluation conditions --------------------------------------------------
def nominal_eval():
    return sim.SensingParams(slip=0.92, gyro_bias=0.06, sig_r=0.12, sig_b=0.06,
                             dropout_len=14, dropout_period=45, n_landmarks=6)


AXES = {
    "dropout_len":  [4, 10, 16, 22, 28, 34],
    "sig_r":        [0.05, 0.12, 0.20, 0.30, 0.40, 0.50],
    "n_landmarks":  [2, 3, 4, 5, 6, 8],
    "gyro_bias":    [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
}


def _apply(base, axis, val):
    from dataclasses import replace
    if axis == "sig_r":
        return replace(base, sig_r=val, sig_b=val * 0.5)
    return replace(base, **{axis: val})


# ---- closed-loop evaluation of one observer under one condition -------------
def eval_closed(name, trained, params_sim, seeds):
    ref = sim.figure_eight(N_STEPS, params_sim.dt, A=A_PATH)
    ms = []
    for sd in seeds:
        rng = np.random.default_rng(1000 + sd)
        active = sim.active_mask(params_sim.n_landmarks, rng, randomize=True)
        est = build(name, trained, params_sim)
        ctrl = controllers.PurePursuit(ref[0], v_cmd=1.0, Ld=0.8)
        out = sim.rollout(ctrl, est, params_sim, sim.FIXED_MAP, ref, rng, active=active)
        ms.append(metrics.all_metrics(out, params_sim.dt))
    return ms


def eval_open(name, trained, params_sim, seeds):
    """Open-loop replay RMSE on expert-generated data (identical inputs to all
    observers per seed)."""
    import copy
    res = []
    for sd in seeds:
        rng = np.random.default_rng(5000 + sd)
        cond = copy.copy(params_sim)
        cond.n_landmarks = params_sim.n_landmarks
        traj = data_mod.expert_rollout(cond, N_STEPS, rng)
        est = build(name, trained, params_sim)
        est.reset(traj["s0"])
        E = np.stack([est.step(traj["Uodo"][k], traj["Z"][k], traj["M"][k], k)
                      for k in range(N_STEPS)])
        res.append((metrics.pos_rmse(traj["S"], E), metrics.head_rmse(traj["S"], E)))
    return res
