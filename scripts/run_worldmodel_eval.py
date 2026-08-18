"""World-model extension: full evaluation pipeline.

Produces (under results/worldmodel/):
  wm_nominal.json           closed+open-loop metrics, nominal condition,
                             CORE observers + imagination-curriculum arms
  wm_sweep_<axis>.json      same, across each of the 4 degradation sweeps
  wm_compound.json          compound-stress condition
  imagination_nominal.json  imagine_rmse / imagine_auc, all observers, nominal
  imagination_sweeps.json   imagine_rmse across the dropout_len sweep (the
                             regime the imagination curriculum targets)
  proxy_with_imagination.json  extended offline-selection-criteria table
                                (Table 3 + imagine_rmse as a 6th proxy)
  seed_robustness_imagine.json seed-to-seed variation of ssm_ekf_imagine
"""
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from ssm_obs import sim, controllers, metrics, persist, data as data_mod
from ssm_obs import experiments as E
from ssm_obs import imagination as im
from ssm_obs import controller_metrics as cm

RES = os.path.join(os.path.dirname(__file__), "..", "results", "worldmodel")
os.makedirs(RES, exist_ok=True)
NSEED = int(os.environ.get("NSEED", 10))
SEEDS = list(range(NSEED))

WM_MODELDIR = os.path.join(os.path.dirname(__file__), "..", "results", "models_worldmodel")
WM_NAMES = ["ssm_ekf_imagine_seed0", "ssm_dr_imagine_seed0", "gru_ekf_imagine_seed0"]
WM_SEED_NAMES = ["ssm_ekf_imagine_seed0", "ssm_ekf_imagine_seed1", "ssm_ekf_imagine_seed2"]

PRETTY_WM = {
    "ssm_ekf_imagine_seed0": "SSM-EKF (imagination-trained)",
    "ssm_dr_imagine_seed0": "SSM-DR (imagination-trained)",
    "gru_ekf_imagine_seed0": "GRU-EKF (imagination-trained)",
}

CORE = E.CORE  # dead_reckoning, ekf, gru_dr, ssm_dr, ssm_ekf, gru_ekf
ALL_MAIN = CORE + WM_NAMES


def load_trained():
    trained = {}
    for spec in E.ZOO:
        s, w = persist.load(os.path.join("results", "models", spec.name))
        trained[spec.name] = (s, w)
    for name in WM_SEED_NAMES:
        s, w = persist.load(os.path.join(WM_MODELDIR, name))
        trained[name] = (s, w)
    for name in WM_NAMES:
        if name not in trained:
            s, w = persist.load(os.path.join(WM_MODELDIR, name))
            trained[name] = (s, w)
    return trained


def agg(ms, key):
    v = np.array([m[key] for m in ms])
    return float(v.mean()), float(v.std() / max(1, len(v)) ** 0.5 * 1.96)


def dump(name, obj):
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote results/worldmodel/{name}")


# ---------------------------------------------------------------------------
def eval_row(name, trained, ps, seeds):
    cl = E.eval_closed(name, trained, ps, seeds)
    ol = E.eval_open(name, trained, ps, seeds)
    row = {k: agg(cl, k) for k in ["pos_rmse", "head_rmse", "ct_rmse", "ct_max",
                                    "effort", "recovery", "diverge"]}
    opr = np.array([r[0] for r in ol]); ohr = np.array([r[1] for r in ol])
    row["ol_pos_rmse"] = [float(opr.mean()), float(opr.std() / len(opr) ** 0.5 * 1.96)]
    row["ol_head_rmse"] = [float(ohr.mean()), float(ohr.std() / len(ohr) ** 0.5 * 1.96)]
    return row


def do_nominal(trained):
    ps = E.nominal_eval()
    out = {}
    for name in ALL_MAIN:
        row = eval_row(name, trained, ps, SEEDS)
        out[name] = row
        print(f"  {name:28s} clPos={row['pos_rmse'][0]:.3f} ct={row['ct_rmse'][0]:.3f} "
              f"olPos={row['ol_pos_rmse'][0]:.3f}")
    dump("wm_nominal.json", out)
    return out


def do_axis(axis, trained):
    base = E.nominal_eval()
    out = {n: [] for n in ALL_MAIN}
    out["_levels"] = E.AXES[axis]
    for val in E.AXES[axis]:
        ps = E._apply(base, axis, val)
        for name in ALL_MAIN:
            row = eval_row(name, trained, ps, SEEDS)
            out[name].append(row)
        print(f"  {axis}={val}: done")
    dump(f"wm_sweep_{axis}.json", out)
    return out


def do_compound(trained):
    base = E.nominal_eval()
    from dataclasses import replace
    ps = replace(base, slip=0.85, gyro_bias=0.18, sig_r=0.35, sig_b=0.18,
                 dropout_len=26, dropout_period=45, n_landmarks=3)
    out = {}
    for name in ALL_MAIN:
        row = eval_row(name, trained, ps, SEEDS)
        out[name] = row
        print(f"  compound {name:28s} clPos={row['pos_rmse'][0]:.3f} ct={row['ct_rmse'][0]:.3f}")
    dump("wm_compound.json", out)
    return out


# ---------------------------------------------------------------------------
def eval_imagination(name, trained, ps, seeds, spec_im):
    """imagine_rmse / imagine_auc for one observer under one condition,
    aggregated over `seeds` independent replay trajectories (same protocol
    as eval_open: a fresh expert-controlled trajectory per seed)."""
    rmses, aucs = [], []
    for sd in seeds:
        rng = np.random.default_rng(5000 + sd)
        cond = copy.copy(ps)
        traj = data_mod.expert_rollout(cond, E.N_STEPS, rng)
        out = dict(S=traj["S"], Uodo=traj["Uodo"], Z=traj["Z"], M=traj["M"])
        est = E.build(name, trained, ps)
        res = im.imagination_rollout(est, out, spec_im)
        s = im.imagination_summary(res)
        if not np.isnan(s["imagine_rmse"]):
            rmses.append(s["imagine_rmse"]); aucs.append(s["imagine_auc"])
    return float(np.mean(rmses)), float(np.mean(aucs))


def do_imagination_nominal(trained):
    ps = E.nominal_eval()
    spec_im = im.ImaginationSpec(horizon=20, stride=10, warmup=10)
    out = {}
    for name in ALL_MAIN:
        r, a = eval_imagination(name, trained, ps, SEEDS, spec_im)
        out[name] = dict(imagine_rmse=r, imagine_auc=a)
        print(f"  {name:28s} imagine_rmse={r:.3f} imagine_auc={a:.3f}")
    dump("imagination_nominal.json", out)
    return out


def do_imagination_dropout_sweep(trained):
    """The regime the curriculum targets: does imagination-training reduce
    blind-rollout error, and does that translate to closed-loop robustness,
    specifically as blackouts get longer?"""
    base = E.nominal_eval()
    spec_im = im.ImaginationSpec(horizon=20, stride=8, warmup=8)
    levels = E.AXES["dropout_len"]
    names = ["dead_reckoning", "ekf", "ssm_ekf", "ssm_ekf_imagine_seed0",
             "ssm_dr", "ssm_dr_imagine_seed0"]
    out = {n: [] for n in names}
    out["_levels"] = levels
    for val in levels:
        ps = E._apply(base, "dropout_len", val)
        for name in names:
            r, a = eval_imagination(name, trained, ps, SEEDS, spec_im)
            out[name].append(dict(imagine_rmse=r, imagine_auc=a))
        print(f"  dropout_len={val}: done")
    dump("imagination_sweeps.json", out)
    return out


# ---------------------------------------------------------------------------
def do_proxy_with_imagination(trained, nominal_res, sweep_results):
    """Extend the paper's Table 3 (offline selection criteria) with
    imagine_rmse as a 6th candidate proxy, using the SAME 24 conditions
    (6 levels x 4 axes) and the SAME 6 CORE observers, so it is directly
    comparable to the audited Table 3 in the original report."""
    base = E.nominal_eval()
    spec_im = im.ImaginationSpec(horizon=20, stride=10, warmup=10)

    conditions, proxy_pose, proxy_head, proxy_imagine, closed = [], {}, {}, {}, {}
    for axis, levels in E.AXES.items():
        for val in levels:
            cond_name = f"{axis}={val}"
            conditions.append(cond_name)
            ps = E._apply(base, axis, val)
            proxy_pose[cond_name] = {}
            proxy_head[cond_name] = {}
            proxy_imagine[cond_name] = {}
            closed[cond_name] = {}
            for name in CORE:
                cl = E.eval_closed(name, trained, ps, SEEDS)
                ol = E.eval_open(name, trained, ps, SEEDS)
                closed[cond_name][name] = float(np.mean([m["ct_rmse"] for m in cl]))
                proxy_pose[cond_name][name] = float(np.mean([r[0] for r in ol]))
                proxy_head[cond_name][name] = float(np.mean([r[1] for r in ol]))
                r_im, _ = eval_imagination(name, trained, ps, SEEDS, spec_im)
                proxy_imagine[cond_name][name] = r_im
        print(f"  proxy axis {axis}: done")

    def spearman(x, y):
        from scipy.stats import spearmanr
        r, _ = spearmanr(x, y)
        return float(r)

    flat_pose, flat_head, flat_imagine, flat_closed = [], [], [], []
    for c in conditions:
        for name in CORE:
            flat_pose.append(proxy_pose[c][name])
            flat_head.append(proxy_head[c][name])
            flat_imagine.append(proxy_imagine[c][name])
            flat_closed.append(closed[c][name])

    summary_pose = cm.selection_summary(conditions, CORE, proxy_pose, closed)
    summary_head = cm.selection_summary(conditions, CORE, proxy_head, closed)
    summary_imagine = cm.selection_summary(conditions, CORE, proxy_imagine, closed)

    out = {
        "n_conditions": len(conditions),
        "position_rmse": {
            "spearman": spearman(flat_pose, flat_closed),
            "flip_fraction": summary_pose["flip_fraction"],
            "mean_regret": summary_pose["mean_regret"],
            "max_regret": summary_pose["max_regret"],
        },
        "heading_rmse": {
            "spearman": spearman(flat_head, flat_closed),
            "flip_fraction": summary_head["flip_fraction"],
            "mean_regret": summary_head["mean_regret"],
            "max_regret": summary_head["max_regret"],
        },
        "imagine_rmse_H20": {
            "spearman": spearman(flat_imagine, flat_closed),
            "flip_fraction": summary_imagine["flip_fraction"],
            "mean_regret": summary_imagine["mean_regret"],
            "max_regret": summary_imagine["max_regret"],
        },
        "raw": {"conditions": conditions, "proxy_pose": proxy_pose, "proxy_head": proxy_head,
                "proxy_imagine": proxy_imagine, "closed": closed},
    }
    dump("proxy_with_imagination.json", out)
    return out


def do_seed_robustness(trained):
    ps = E.nominal_eval()
    conds = {
        "nominal": ps,
        "long_blackout": E._apply(ps, "dropout_len", 34),
        "high_noise": E._apply(ps, "sig_r", 0.5),
        "two_landmarks": E._apply(ps, "n_landmarks", 2),
        "high_bias": E._apply(ps, "gyro_bias", 0.25),
    }
    out = {}
    for cname, cps in conds.items():
        out[cname] = {}
        for name in WM_SEED_NAMES:
            cl = E.eval_closed(name, trained, cps, SEEDS)
            out[cname][name] = agg(cl, "ct_rmse")
        print(f"  seed robustness {cname}: done")
    dump("seed_robustness_imagine.json", out)
    return out


def main():
    t0 = time.time()
    trained = load_trained()
    print("[nominal]"); nominal_res = do_nominal(trained)
    sweep_results = {}
    for axis in E.AXES:
        print(f"[sweep {axis}]"); sweep_results[axis] = do_axis(axis, trained)
    print("[compound]"); do_compound(trained)
    print("[imagination nominal]"); do_imagination_nominal(trained)
    print("[imagination dropout sweep]"); do_imagination_dropout_sweep(trained)
    print("[proxy + imagination]"); do_proxy_with_imagination(trained, nominal_res, sweep_results)
    print("[seed robustness]"); do_seed_robustness(trained)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
