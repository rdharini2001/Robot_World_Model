#!/usr/bin/env python3
"""Real-data and MRCLAM-calibrated replay experiments."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","src"))
import numpy as np
from scipy.stats import spearmanr
from ssm_obs import controllers, controller_metrics as cm, dynamics as dyn, ekf, metrics, mrclam, sim, models, persist

ROOT=Path(__file__).resolve().parents[1]
Q=[0.25,1.0,4.0];R=[0.25,1.0,4.0]
NAMES=["dead_reckoning"]+[f"ekf_q{q:g}_r{r:g}" for q in Q for r in R]


def observer(name,p,lm):
    if name=="dead_reckoning":return ekf.DeadReckoning(p)
    q=float(name.split("_q")[1].split("_r")[0]);r=float(name.split("_r")[1])
    return ekf.EKF(p,lm,q_scale=q,r_scale=r)


def run_log(name,p,log):
    o=observer(name,p,log["landmarks"]);o.reset(log["s0"]);E=[]
    for k in range(len(log["S"])):E.append(o.step(log["Uodo"][k],log["Z"][k],log["M"][k],k))
    return np.asarray(E)


def ref_from_S(S,dt):
    th=np.unwrap(S[:,2]);dx=np.gradient(S[:,0],dt);dy=np.gradient(S[:,1],dt)
    return S[:,:2],dyn.wrap(th),dx*np.cos(th)+dy*np.sin(th),np.gradient(th,dt)


def params_from_audit(a,dt):
    ov,ow=a["odometry_robust_std"];mr,mb=a["measurement_robust_std"]
    return sim.SensingParams(dt=dt,sig_v=max(ov,0.01),sig_w=max(ow,0.01),
        sig_r=max(mr,0.03),sig_b=max(mb,0.02),sensor_range=8.0,slip=1.0,gyro_bias=0.0,
        dropout_len=0,dropout_period=100000,n_landmarks=a["n_landmarks"])


def empirical_expert_log(p,lm,ref,noise,rng):
    P,th,v,w=ref;n=len(P);s=np.array([P[0,0],P[0,1],th[0]])
    S=np.zeros((n,3));Uodo=np.zeros((n,2));Z=np.zeros((n,len(lm),2));M=np.zeros((n,len(lm)))
    ctrl=controllers.PurePursuit(P,1.0,.8)
    counts=noise.measurement_counts;start=int(rng.integers(0,max(1,len(counts)-n)));seq=np.resize(counts[start:start+n],n)
    for k in range(n):
        u=np.clip(ctrl(s,k),[0,-3],[2.5,3]);oe=noise.odometry_residuals[rng.integers(len(noise.odometry_residuals))]
        Uodo[k]=u+oe;dist=np.linalg.norm(lm-s[:2],axis=1);cand=np.where(dist<=p.sensor_range)[0]
        c=min(int(seq[k]),len(cand))
        if c:
            weights=np.exp(-dist[cand]/p.sensor_range);weights/=weights.sum();chosen=rng.choice(cand,c,False,p=weights)
            for j in chosen:
                me=noise.measurement_residuals[rng.integers(len(noise.measurement_residuals))]
                Z[k,j]=dyn.meas_predict(s,lm[j])+me;Z[k,j,1]=dyn.wrap(Z[k,j,1]);M[k,j]=1
        S[k]=s;s=dyn.step_true(s,u,p.dt)
    return {"S":S,"Uodo":Uodo,"Z":Z,"M":M,"s0":S[0],"landmarks":lm,"dt":p.dt}


def summarize_selection(proxy,closed):
    rows={};obs=list(closed)
    for f in next(iter(proxy.values())).keys():
        pick=min(obs,key=lambda n:proxy[n][f]);oracle=min(obs,key=lambda n:closed[n])
        rows[f]={"chosen":pick,"oracle":oracle,"regret":closed[pick]-closed[oracle],
                 "spearman":float(spearmanr([proxy[n][f] for n in obs],[closed[n] for n in obs]).statistic)}
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data-dir",required=True);ap.add_argument("--robot",type=int,default=1)
    ap.add_argument("--duration",type=float,default=600);ap.add_argument("--seeds",type=int,default=20);ap.add_argument("--replay-steps",type=int,default=800);a=ap.parse_args()
    log=mrclam.load_robot(a.data_dir,a.robot,dt=.1,duration=a.duration);audit=mrclam.audit_noise(log);p=params_from_audit(audit,log["dt"])
    pp=lambda:controllers.TimeIndexedPurePursuit(log["S"][:,:2],1.0,.8)
    kan_ref=ref_from_S(log["S"],log["dt"]);kan=lambda:controllers.Kanayama(kan_ref)
    raw={}
    for name in NAMES:
        E=run_log(name,p,log)
        raw[name]={"pose_rmse":metrics.pos_rmse(log["S"],E),"heading_rmse":metrics.head_rmse(log["S"],E),
                   "controller_disagreement":cm.controller_disagreement(log["S"],E,pp),
                   "lcse":cm.local_control_sensitivity_error(log["S"],E,pp),
                   "kanayama_disagreement":cm.controller_disagreement(log["S"],E,kan)}
        print("raw",name,raw[name]["pose_rmse"])

    # Deliberate zero-shot stress test of the learned checkpoints. The original
    # architecture has exactly eight ordered landmark slots, so use an
    # eight-landmark subset and apply the synthetic-map checkpoints unchanged.
    log8=mrclam.load_robot(a.data_dir,a.robot,dt=.1,duration=a.duration,max_landmarks=8)
    audit8=mrclam.audit_noise(log8);p8=params_from_audit(audit8,log8["dt"]);p8.n_landmarks=8
    pp8=lambda:controllers.TimeIndexedPurePursuit(log8["S"][:,:2],1.0,.8)
    zero_shot={}
    for name in ["dead_reckoning","ekf","ssm_ekf","gru_ekf"]:
        if name=="dead_reckoning": o=ekf.DeadReckoning(p8)
        elif name=="ekf": o=ekf.EKF(p8,log8["landmarks"],q_scale=.25,r_scale=4.0)
        else:
            spec,w=persist.load(str(ROOT/"results"/"models"/name))
            o=models.LearnedObserver(w,spec.cell,spec.anchor,p8,log8["landmarks"],
                use_mask=spec.use_mask,selective=spec.selective,use_cov=spec.use_cov)
        o.reset(log8["s0"]); E8=[]
        for k in range(len(log8["S"])): E8.append(o.step(log8["Uodo"][k],log8["Z"][k],log8["M"][k],k))
        E8=np.asarray(E8)
        zero_shot[name]={"pose_rmse":metrics.pos_rmse(log8["S"],E8),
            "heading_rmse":metrics.head_rmse(log8["S"],E8),
            "controller_disagreement":cm.controller_disagreement(log8["S"],E8,pp8),
            "lcse":cm.local_control_sensitivity_error(log8["S"],E8,pp8)}

    # MRCLAM-calibrated closed-loop replay on the physical landmark geometry,
    # recentered so the reference is covered by the map.
    noise=mrclam.fit_empirical_noise(log,center=False);lm=log["landmarks"]-log["landmarks"].mean(axis=0)
    ref=sim.figure_eight(a.replay_steps,p.dt,A=2.0);closed_seed={n:[] for n in NAMES};proxy_seed={n:[] for n in NAMES}
    for sd in range(a.seeds):
        # Common logged expert inputs for all candidates.
        expert=empirical_expert_log(p,lm,ref,noise,np.random.default_rng(7100+sd))
        pp_e=lambda:controllers.TimeIndexedPurePursuit(expert["S"][:,:2],1.0,.8)
        for name in NAMES:
            o=observer(name,p,lm);o.reset(expert["s0"]);E=[]
            for k in range(len(expert["S"])):E.append(o.step(expert["Uodo"][k],expert["Z"][k],expert["M"][k],k))
            E=np.asarray(E)
            proxy_seed[name].append({"pose_rmse":metrics.pos_rmse(expert["S"],E),
                "controller_disagreement":cm.controller_disagreement(expert["S"],E,pp_e),
                "lcse":cm.local_control_sensitivity_error(expert["S"],E,pp_e)})
            out=mrclam.rollout_empirical(controllers.PurePursuit(ref[0],1.0,.8),observer(name,p,lm),p,lm,ref,noise,np.random.default_rng(9100+sd))
            closed_seed[name].append(metrics.cross_track_rmse(out["S"],out["P"]))
    fields=list(proxy_seed[NAMES[0]][0]);proxy={n:{f:float(np.mean([x[f] for x in proxy_seed[n]])) for f in fields} for n in NAMES}
    closed={n:float(np.mean(closed_seed[n])) for n in NAMES}
    replay={"proxy_mean":proxy,"closed_loop_ct_mean":closed,"selection":summarize_selection(proxy,closed),
            "closed_loop_seed_values":closed_seed,"proxy_seed_values":proxy_seed}
    out={"dataset":"UTIAS MRCLAM Dataset 1","robot":a.robot,"audit":audit,"filter_params":p.__dict__,
         "raw_log_localization":raw,"zero_shot_learned_8_landmarks":zero_shot,
         "zero_shot_note":"The learned checkpoints are applied without retraining to an 8-landmark subset. Their ordered landmark-slot representation was trained on one synthetic map, making this an intentionally hard out-of-domain stress test.",
         "empirical_replay":replay,
         "scope_note":"Raw localization uses physical robot logs. Closed-loop replay uses simulated dynamics with odometry, measurement residuals, and observation timing resampled from the physical logs."}
    d=ROOT/"results"/"enhanced";d.mkdir(parents=True,exist_ok=True);json.dump(out,open(d/"mrclam_results.json","w"),indent=2)
    print("\nreplay oracle",min(closed,key=closed.get),closed[min(closed,key=closed.get)])
    for f,r in replay["selection"].items():print(f,r)

if __name__=="__main__":main()
