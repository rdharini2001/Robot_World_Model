import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from ssm_obs import persist, sim, controllers, plotting
from ssm_obs import experiments as E
from ssm_obs.experiments import ZOO, CORE, N_STEPS, A_PATH

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)

def load_zoo():
    return {s.name: persist.load(os.path.join(RES, "models", s.name)) for s in ZOO}

def capture_timeseries(trained):
    """One hard-ish rollout; log per-step position error for three observers."""
    from dataclasses import replace
    ps = replace(E.nominal_eval(), dropout_len=24, dropout_period=40, gyro_bias=0.12, sig_r=0.2)
    ref = sim.figure_eight(N_STEPS, ps.dt, A=A_PATH)
    names = ["dead_reckoning", "ekf", "ssm_ekf"]
    series = {}; on_ref = None
    for n in names:
        rng = np.random.default_rng(7)                       # SAME seed -> same world
        active = sim.active_mask(ps.n_landmarks, rng, randomize=True)
        est = E.build(n, trained, ps)
        ctrl = controllers.PurePursuit(ref[0], v_cmd=1.0, Ld=0.8)
        out = sim.rollout(ctrl, est, ps, sim.FIXED_MAP, ref, rng, active=active)
        err = np.linalg.norm(out["S"][:, :2] - out["E"][:, :2], axis=1)
        series[n] = err.tolist(); on_ref = out["on"].tolist()
    json.dump({"on": on_ref, "dt": ps.dt, "series": series},
              open(os.path.join(RES, "timeseries.json"), "w"))
    return {"on": on_ref, "dt": ps.dt, "series": series}

def capture_trajectory(trained):
    ps = E.nominal_eval()
    ref = sim.figure_eight(N_STEPS, ps.dt, A=A_PATH)
    names = ["dead_reckoning", "ekf", "ssm_ekf"]
    traj = {}
    for n in names:
        rng = np.random.default_rng(3)
        active = sim.active_mask(ps.n_landmarks, rng, randomize=True)
        est = E.build(n, trained, ps)
        ctrl = controllers.PurePursuit(ref[0], v_cmd=1.0, Ld=0.8)
        out = sim.rollout(ctrl, est, ps, sim.FIXED_MAP, ref, rng, active=active)
        traj[n] = out["S"][:, :2].tolist()
    payload = {"P": ref[0].tolist(), "landmarks": sim.FIXED_MAP.tolist(), "traj": traj}
    json.dump(payload, open(os.path.join(RES, "trajectory.json"), "w"))
    return payload

def main():
    trained = load_zoo()
    print("capturing time-series..."); ts = capture_timeseries(trained)
    print("capturing trajectory..."); trj = capture_trajectory(trained)
    print("rendering figures...")
    plotting.fig_sweeps(RES, FIG, CORE)
    plotting.fig_mismatch(RES, FIG)
    plotting.fig_timeseries(ts, FIG)
    plotting.fig_trajectory(trj, FIG)
    plotting.fig_ablation(RES, FIG)
    print("figures written to", FIG)
    for f in sorted(os.listdir(FIG)): print("  ", f)

if __name__ == "__main__":
    main()
