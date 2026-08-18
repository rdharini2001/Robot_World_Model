#!/usr/bin/env python3
"""Matched EKF-anchor ablations for selectivity and mask awareness."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","src"))
import numpy as np
from ssm_obs import experiments as E, persist

ROOT=Path(__file__).resolve().parents[1]
NAMES=E.ABLATIONS


def load():
    return {n:persist.load(str(ROOT/"results"/"models"/n)) for n in NAMES}


def mean_ci(vals):
    a=np.asarray(vals,float);return [float(a.mean()),float(1.96*a.std(ddof=1)/np.sqrt(len(a)))]


def paired_bootstrap(base,other,n=5000,seed=0):
    base=np.asarray(base);other=np.asarray(other);delta=other-base;rng=np.random.default_rng(seed)
    means=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(n)])
    return {"mean_delta":float(delta.mean()),"ci95":np.quantile(means,[.025,.975]).tolist(),
            "p_delta_le_zero":float(np.mean(means<=0))}


def condition(name,p,trained,seeds):
    cl={n:E.eval_closed(n,trained,p,seeds) for n in NAMES};ol={n:E.eval_open(n,trained,p,seeds) for n in NAMES}
    out={}
    for n in NAMES:
        ct=[x["ct_rmse"] for x in cl[n]];div=[x["diverge"] for x in cl[n]];opr=[x[0] for x in ol[n]]
        out[n]={"ct_rmse":mean_ci(ct),"divergence":mean_ci(div),"open_pose_rmse":mean_ci(opr),
                "ct_seed":ct,"open_pose_seed":opr}
    for n in NAMES[1:]:out[n]["paired_vs_selective_ct"]=paired_bootstrap(out[NAMES[0]]["ct_seed"],out[n]["ct_seed"])
    return out


def main():
    trained=load();seeds=list(range(10))
    selected={
        "nominal":E.nominal_eval(),
        "long_blackout":E._apply(E.nominal_eval(),"dropout_len",34),
        "high_range_noise":E._apply(E.nominal_eval(),"sig_r",0.5),
        "two_landmarks":E._apply(E.nominal_eval(),"n_landmarks",2),
        "high_gyro_bias":E._apply(E.nominal_eval(),"gyro_bias",0.25),
    }
    res={"models":NAMES,"conditions":{}}
    out=ROOT/"results"/"enhanced"/"matched_ekf_ablation.json"
    for label,p in selected.items():
        print(label,flush=True);res["conditions"][label]=condition(label,p,trained,seeds)
        json.dump(res,open(out,"w"),indent=2)
    for n in NAMES:
        m=res["conditions"]["nominal"][n];print(n,"ct",m["ct_rmse"],"ol",m["open_pose_rmse"],"div",m["divergence"])

if __name__=="__main__":main()
