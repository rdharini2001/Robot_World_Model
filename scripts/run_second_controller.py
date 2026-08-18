#!/usr/bin/env python3
"""Secondary-controller sensitivity using Kanayama feedback.

This intentionally lean script reuses the audited 10-seed replay proxy and
runs matched pure-pursuit and Kanayama closed-loop sensitivity experiments with
a configurable number of evaluation seeds. It tests whether the observer chosen
by replay RMSE and the closed-loop oracle are controller-dependent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssm_obs import controller_metrics as cm
from ssm_obs import controllers, metrics, persist, sim
from ssm_obs import experiments as E

ROOT = Path(__file__).resolve().parents[1]
OBSERVERS = E.CORE


def load_models():
    trained = {}
    for name in OBSERVERS:
        if name not in ("dead_reckoning", "ekf"):
            trained[name] = persist.load(str(ROOT / "results" / "models" / name))
    return trained



def analyze_pose_proxy(conditions, pose_proxy, closed):
    proxy_flat = np.asarray([
        pose_proxy[c][o] for c in conditions for o in OBSERVERS
    ])
    closed_flat = np.asarray([
        closed[c][o] for c in conditions for o in OBSERVERS
    ])
    return {
        "spearman": float(spearmanr(proxy_flat, closed_flat).statistic),
        "pearson": float(pearsonr(proxy_flat, closed_flat).statistic),
        "selection": cm.selection_summary(conditions, OBSERVERS, pose_proxy, closed),
    }


def paired_seed_bootstrap(seed_rows, conditions, pose_proxy, n_boot=5000, seed=260726):
    values = {}
    seed_ids = sorted({row["seed"] for row in seed_rows})
    for row in seed_rows:
        key = (row["controller"], row["condition"], row["observer"])
        values.setdefault(key, {})[row["seed"]] = row["cross_track_rmse"]
    rng = np.random.default_rng(seed)
    kanayama_failure = []
    kanayama_regret = []
    oracle_change = []
    for _ in range(n_boot):
        sample = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        closed = {
            controller_name: {
                condition: {
                    observer_name: float(np.mean([
                        values[(controller_name, condition, observer_name)][int(seed_id)]
                        for seed_id in sample
                    ]))
                    for observer_name in OBSERVERS
                }
                for condition in conditions
            }
            for controller_name in ("pure_pursuit", "kanayama")
        }
        selection = cm.selection_summary(
            conditions, OBSERVERS, pose_proxy, closed["kanayama"]
        )
        kanayama_failure.append(selection["flip_fraction"])
        kanayama_regret.append(selection["mean_regret"])
        oracle_change.append(np.mean([
            min(OBSERVERS, key=lambda o: closed["pure_pursuit"][c][o])
            != min(OBSERVERS, key=lambda o: closed["kanayama"][c][o])
            for c in conditions
        ]))
    return {
        "n_bootstrap": n_boot,
        "kanayama_pose_selection_failure_ci95": np.quantile(
            kanayama_failure, [0.025, 0.975]
        ).tolist(),
        "kanayama_pose_mean_regret_ci95": np.quantile(
            kanayama_regret, [0.025, 0.975]
        ).tolist(),
        "controller_oracle_change_fraction_ci95": np.quantile(
            oracle_change, [0.025, 0.975]
        ).tolist(),
        "note": (
            "Paired bootstrap over the matched evaluation seeds; with only five "
            "seeds by default, these intervals are sensitivity estimates."
        ),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    trained = load_models()
    proxy_result = json.loads(
        (ROOT / "results" / "enhanced" / "proxy_analysis.json").read_text()
    )
    pose_proxy = proxy_result["proxy_means"]["pose_rmse"]
    matched_closed = {"pure_pursuit": {}, "kanayama": {}}
    seed_rows = []
    for axis, levels in E.AXES.items():
        for value in levels:
            condition = f"{axis}={value}"
            params = E._apply(E.nominal_eval(), axis, value)
            ref = sim.figure_eight(E.N_STEPS, params.dt, A=E.A_PATH)
            for controller_name in matched_closed:
                matched_closed[controller_name][condition] = {}
                for observer_name in OBSERVERS:
                    values = []
                    for seed in range(args.seeds):
                        rng = np.random.default_rng(21000 + seed)
                        active = sim.active_mask(params.n_landmarks, rng, randomize=True)
                        observer = E.build(observer_name, trained, params)
                        controller = (
                            controllers.PurePursuit(ref[0], v_cmd=1.0, Ld=0.8)
                            if controller_name == "pure_pursuit"
                            else controllers.Kanayama(ref)
                        )
                        rollout = sim.rollout(
                            controller,
                            observer,
                            params,
                            sim.FIXED_MAP,
                            ref,
                            rng,
                            active=active,
                        )
                        value_ct = float(metrics.cross_track_rmse(rollout["S"], rollout["P"]))
                        values.append(value_ct)
                        seed_rows.append({
                            "condition": condition,
                            "axis": axis,
                            "level": value,
                            "controller": controller_name,
                            "observer": observer_name,
                            "seed": seed,
                            "cross_track_rmse": value_ct,
                        })
                    matched_closed[controller_name][condition][observer_name] = float(np.mean(values))
            print("finished", condition, flush=True)

    pure_pursuit_matched = matched_closed["pure_pursuit"]
    kanayama = matched_closed["kanayama"]
    conditions = list(kanayama)
    oracle_rows = []
    for condition in conditions:
        pp_oracle = min(OBSERVERS, key=lambda o: pure_pursuit_matched[condition][o])
        ka_oracle = min(OBSERVERS, key=lambda o: kanayama[condition][o])
        oracle_rows.append({
            "condition": condition,
            "pure_pursuit_oracle": pp_oracle,
            "kanayama_oracle": ka_oracle,
            "changed": pp_oracle != ka_oracle,
        })

    result = {
        "observers": OBSERVERS,
        "kanayama_evaluation_seeds": args.seeds,
        "pose_proxy_source": "proxy_analysis.json (10 common-input replay seeds)",
        "pure_pursuit_reference_source": "proxy_analysis.json / original 10-seed sweeps",
        "matched_closed_loop": matched_closed,
        "kanayama_closed_loop": kanayama,
        "pose_rmse_vs_pure_pursuit_matched": analyze_pose_proxy(
            conditions, pose_proxy, pure_pursuit_matched
        ),
        "pose_rmse_vs_kanayama": analyze_pose_proxy(
            conditions, pose_proxy, kanayama
        ),
        "controller_oracle_transfer": {
            "n_changed": int(sum(row["changed"] for row in oracle_rows)),
            "n_conditions": len(oracle_rows),
            "change_fraction": float(np.mean([row["changed"] for row in oracle_rows])),
            "rows": oracle_rows,
        },
        "scope_note": (
            "This is a secondary synthetic sensitivity analysis. Pure pursuit and "
            "Kanayama use identical evaluation seeds, plant, checkpoints, conditions, "
            "and actuator limits, but only the stated number of seeds."
        ),
    }
    result["matched_seed_bootstrap"] = paired_seed_bootstrap(
        seed_rows, conditions, pose_proxy
    )
    out_dir = ROOT / "results" / "enhanced"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "second_controller_kanayama.json").write_text(json.dumps(result, indent=2))
    with (out_dir / "second_controller_kanayama_seed.jsonl").open("w") as handle:
        for row in seed_rows:
            handle.write(json.dumps(row) + "\n")

    print("\nPose RMSE vs Kanayama:", result["pose_rmse_vs_kanayama"])
    print("Controller oracle transfer:", result["controller_oracle_transfer"]["n_changed"],
          "/", result["controller_oracle_transfer"]["n_conditions"])


if __name__ == "__main__":
    main()
