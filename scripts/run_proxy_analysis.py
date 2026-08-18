#!/usr/bin/env python3
"""Evaluate offline selection proxies against closed-loop tracking.

This script fixes the original five-observer analysis, includes GRU-EKF, keeps
seed-level measurements, reports top-1 selection regret, and bootstraps the
comparison between pose RMSE and controller-aware proxies.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from scipy.stats import pearsonr, spearmanr
from ssm_obs import controllers, controller_metrics as cm, data as data_mod, metrics, persist, sim
from ssm_obs import experiments as E

ROOT = Path(__file__).resolve().parents[1]
MAIN = E.CORE


def load_models():
    out = {}
    specs = {s.name: s for s in E.ZOO}
    for name in MAIN:
        if name in ("dead_reckoning", "ekf"):
            continue
        path = ROOT / "results" / "models" / name
        if not path.with_suffix(".npz").exists():
            raise FileNotFoundError(f"Missing checkpoint {path}.npz")
        out[name] = persist.load(str(path))
    return out


def reference_from_states(S, dt):
    P = np.asarray(S[:, :2])
    th = np.unwrap(np.asarray(S[:, 2]))
    dx = np.gradient(P[:, 0], dt); dy = np.gradient(P[:, 1], dt)
    v = dx * np.cos(th) + dy * np.sin(th)
    w = np.gradient(th, dt)
    return P, sim.dyn.wrap(th), v, w


def estimate_log(name, trained, params, traj):
    est = E.build(name, trained, params)
    est.reset(traj["s0"])
    return np.stack([est.step(traj["Uodo"][k], traj["Z"][k], traj["M"][k], k)
                     for k in range(len(traj["S"]))])


def bootstrap_selection(condition_ids, observers, proxy, closed, n_boot=5000, seed=41):
    rng = np.random.default_rng(seed)
    ids = list(condition_ids)
    def one(sample, field):
        flips=[]; regrets=[]
        for c in sample:
            pick=min(observers, key=lambda n: proxy[field][c][n])
            oracle=min(observers, key=lambda n: closed[c][n])
            flips.append(pick != oracle)
            regrets.append(closed[c][pick]-closed[c][oracle])
        return np.mean(flips), np.mean(regrets)
    vals={k:[] for k in proxy}
    for _ in range(n_boot):
        sample=[ids[i] for i in rng.integers(0,len(ids),len(ids))]
        for field in proxy:
            vals[field].append(one(sample, field))
    out={}
    for field, arr in vals.items():
        a=np.asarray(arr)
        out[field]={
            "flip_fraction_ci95": np.quantile(a[:,0],[.025,.975]).tolist(),
            "mean_regret_ci95": np.quantile(a[:,1],[.025,.975]).tolist(),
        }
    # paired observed improvement of the local sensitivity proxy over pose RMSE.
    # With only 24 conditions this interval is descriptive, not definitive.
    rm=np.asarray(vals["pose_rmse"]); task=np.asarray(vals["lcse"])
    diff=rm-task
    out["paired_lcse_improvement"]={
        "flip_reduction_ci95": np.quantile(diff[:,0],[.025,.975]).tolist(),
        "regret_reduction_ci95": np.quantile(diff[:,1],[.025,.975]).tolist(),
        "p_flip_reduction_le_zero": float(np.mean(diff[:,0] <= 0)),
        "p_regret_reduction_le_zero": float(np.mean(diff[:,1] <= 0)),
    }
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=5000)
    args=ap.parse_args()
    trained=load_models()
    raw=[]; proxy_fields=["pose_rmse","heading_rmse","controller_disagreement","lcse","counterfactual_replay"]
    proxy={f:{} for f in proxy_fields}; closed={}; condition_ids=[]
    for axis, levels in E.AXES.items():
        sweep=json.load(open(ROOT/"results"/f"sweep_{axis}.json"))
        for li,val in enumerate(levels):
            cid=f"{axis}={val}"; condition_ids.append(cid)
            params=E._apply(E.nominal_eval(),axis,val)
            closed[cid]={n:float(sweep[n][li]["ct_rmse"][0]) for n in MAIN}
            rows={f:{n:[] for n in MAIN} for f in proxy_fields}
            for sd in range(args.seeds):
                rng=np.random.default_rng(5000+sd)
                traj=data_mod.expert_rollout(params,E.N_STEPS,rng)
                ref=reference_from_states(traj["S"],params.dt)
                pp=lambda P=traj["S"][:,:2]: controllers.TimeIndexedPurePursuit(P,v_cmd=1.0,Ld=0.8)
                # Controller targets/Jacobians depend only on the logged true trajectory,
                # so compute them once and reuse across all observer candidates.
                scale=cm.ControlScale().vector
                pp_controller=pp()
                pp_true=cm.controller_actions(pp_controller,traj["S"])
                pp_j=np.stack([cm.finite_difference_jacobian(pp_controller,s,k) for k,s in enumerate(traj["S"])])
                for name in MAIN:
                    est=estimate_log(name,trained,params,traj)
                    err=cm.state_error(traj["S"],est)
                    pp_est=cm.controller_actions(pp(),est)
                    pp_cd=float(np.sqrt(np.mean(np.sum(((pp_est-pp_true)/scale)**2,axis=1))))
                    pp_lin=np.einsum("tij,tj->ti",pp_j,err)/scale
                    vals={
                        "pose_rmse":metrics.pos_rmse(traj["S"],est),
                        "heading_rmse":metrics.head_rmse(traj["S"],est),
                        "controller_disagreement":pp_cd,
                        "lcse":float(np.sqrt(np.mean(np.sum(pp_lin**2,axis=1)))),
                        "counterfactual_replay":cm.counterfactual_error_replay(
                            traj["S"],est,traj["S"][:,:2],
                            lambda P=traj["S"][:,:2]: controllers.PurePursuit(P,1.0,.8),params.dt),
                    }
                    for f,v in vals.items(): rows[f][name].append(float(v))
                    raw.append({"condition":cid,"axis":axis,"level":val,"seed":sd,
                                "observer":name,**vals,"closed_loop_ct_mean":closed[cid][name]})
            for f in proxy_fields:
                proxy[f][cid]={n:float(np.mean(rows[f][n])) for n in MAIN}
            print(f"finished {cid}")

    analyses={}
    cl_flat=np.array([closed[c][n] for c in condition_ids for n in MAIN])
    for f in proxy_fields:
        x=np.array([proxy[f][c][n] for c in condition_ids for n in MAIN])
        sp=spearmanr(x,cl_flat); pe=pearsonr(x,cl_flat)
        sel=cm.selection_summary(condition_ids,MAIN,proxy[f],closed)
        per_rho=[spearmanr([proxy[f][c][n] for n in MAIN],[closed[c][n] for n in MAIN]).statistic
                 for c in condition_ids]
        analyses[f]={
            "spearman_global":{"rho":float(sp.statistic),"p":float(sp.pvalue)},
            "pearson_global":{"r":float(pe.statistic),"p":float(pe.pvalue)},
            "per_condition_spearman_mean":float(np.nanmean(per_rho)),
            "per_condition_spearman_min":float(np.nanmin(per_rho)),
            "selection":sel,
        }
    boot=bootstrap_selection(condition_ids,MAIN,proxy,closed,args.bootstrap)
    out={"observers":MAIN,"conditions":condition_ids,"proxy_means":proxy,
         "closed_loop_ct":closed,"analysis":analyses,"bootstrap":boot,
         "raw_file":"proxy_seed_level.jsonl"}
    outdir=ROOT/"results"/"enhanced"; outdir.mkdir(parents=True,exist_ok=True)
    json.dump(out,open(outdir/"proxy_analysis.json","w"),indent=2)
    with open(outdir/"proxy_seed_level.jsonl","w") as f:
        for r in raw: f.write(json.dumps(r)+"\n")
    print("\nSUMMARY")
    for f in proxy_fields:
        a=analyses[f]
        print(f"{f:27s} rho={a['spearman_global']['rho']:.3f} "
              f"flip={a['selection']['flip_fraction']:.3f} "
              f"mean_regret={a['selection']['mean_regret']:.4f} "
              f"max_regret={a['selection']['max_regret']:.4f}")


if __name__=="__main__": main()
