#!/usr/bin/env python3
"""Run a small fresh closed-loop experiment using the bundled checkpoints."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from ssm_obs import controllers, experiments as E, metrics, persist, sim

ROOT = Path(__file__).resolve().parents[1]


def load_models():
    trained = {}
    for spec in E.ZOO:
        path = ROOT / "results" / "models" / spec.name
        trained[spec.name] = persist.load(str(path))
    return trained


def main():
    trained = load_models()
    params = E.nominal_eval()
    reference = sim.figure_eight(120, params.dt, A=E.A_PATH)
    names = ["dead_reckoning", "ekf", "gru_ekf", "ssm_ekf"]
    output = {}
    for name in names:
        rng = np.random.default_rng(6562)
        active = sim.active_mask(params.n_landmarks, rng, randomize=True)
        observer = E.build(name, trained, params)
        controller = controllers.PurePursuit(reference[0], v_cmd=1.0, Ld=0.8)
        rollout = sim.rollout(controller, observer, params, sim.FIXED_MAP, reference, rng, active=active)
        row = metrics.all_metrics(rollout, params.dt)
        output[name] = {k: float(v) for k, v in row.items()}

    out_path = ROOT / "results" / "smoke_test.json"
    out_path.write_text(json.dumps(output, indent=2))
    print("Fresh 120-step smoke experiment")
    for name, row in output.items():
        print(f"  {name:16s} cross-track RMSE = {row['ct_rmse']:.3f} m")
    if not all(np.isfinite(row["ct_rmse"]) for row in output.values()):
        raise RuntimeError("Non-finite metric detected")
    if output["ekf"]["ct_rmse"] >= output["dead_reckoning"]["ct_rmse"]:
        raise RuntimeError("Sanity check failed: EKF should improve over dead reckoning")
    print(f"Wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
