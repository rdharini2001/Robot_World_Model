#!/usr/bin/env python3
from pathlib import Path
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"figures"/"enhanced";OUT.mkdir(parents=True,exist_ok=True)

def proxy_plot():
 d=json.load(open(ROOT/"results"/"enhanced"/"proxy_analysis.json"));keys=["pose_rmse","controller_disagreement","lcse","counterfactual_replay"]
 labels=["Pose RMSE","Command disagreement","Local control sensitivity","Counterfactual replay"]
 vals=[100*d["analysis"][k]["selection"]["flip_fraction"] for k in keys]
 fig,ax=plt.subplots(figsize=(7.2,4.4));bars=ax.bar(labels,vals);ax.set_ylabel("Top-1 selection failures (%)");ax.set_title("Offline observer-selection proxies across 24 conditions");ax.tick_params(axis="x",rotation=20)
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v,f"{v:.1f}%",ha="center",va="bottom")
 fig.tight_layout();fig.savefig(OUT/"fig_proxy_failure_rate.png",dpi=220);plt.close(fig)

def ablation_plot():
 d=json.load(open(ROOT/"results"/"enhanced"/"matched_ekf_ablation.json"));conds=["nominal","long_blackout","high_range_noise","two_landmarks","high_gyro_bias"];labs=["Nominal","Long blackout","High range noise","Two landmarks","High gyro bias"]
 models=["ssm_ekf","ssm_ekf_nosel","ssm_ekf_nomask"];mlabs=["Selective + mask","Non-selective + mask","Selective, no mask"];x=np.arange(len(conds));w=.24
 fig,ax=plt.subplots(figsize=(8.2,4.6))
 for i,(m,l) in enumerate(zip(models,mlabs)):ax.bar(x+(i-1)*w,[d["conditions"][c][m]["ct_rmse"][0] for c in conds],w,label=l)
 ax.set_xticks(x,labs,rotation=15);ax.set_ylabel("Closed-loop cross-track RMSE (m)");ax.set_title("Matched ablations on the same EKF anchor");ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/"fig_matched_ekf_ablation.png",dpi=220);plt.close(fig)

def seed_plot():
 d=json.load(open(ROOT/"results"/"enhanced"/"ssm_training_seed_eval.json"));conds=["nominal","long_blackout","high_range_noise","two_landmarks","high_gyro_bias"];labs=["Nominal","Long blackout","High range noise","Two landmarks","High gyro bias"]
 fig,ax=plt.subplots(figsize=(7.8,4.5))
 for sd in ["0","1","2"]:ax.plot(labs,[d["conditions"][c][sd]["mean"] for c in conds],marker="o",label=f"Training seed {sd}")
 ax.set_ylabel("Closed-loop cross-track RMSE (m)");ax.set_title("Selective SSM-EKF sensitivity to training initialization");ax.tick_params(axis="x",rotation=15);ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/"fig_training_seed_robustness.png",dpi=220);plt.close(fig)

def controller_plot():
 d=json.load(open(ROOT/"results"/"enhanced"/"second_controller_kanayama.json"))
 vals=[100*d["pose_rmse_vs_kanayama"]["selection"]["flip_fraction"],100*d["controller_oracle_transfer"]["change_fraction"]]
 labs=["Replay selector failures\n(Kanayama)","Oracle changes\n(Pure pursuit vs. Kanayama)"]
 fig,ax=plt.subplots(figsize=(6.6,4.2));bars=ax.bar(labs,vals);ax.set_ylabel("Conditions (%)");ax.set_ylim(0,max(vals)*1.25);ax.set_title("Secondary five-seed controller sensitivity")
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v,f"{v:.1f}%",ha="center",va="bottom")
 fig.tight_layout();fig.savefig(OUT/"fig_controller_sensitivity.png",dpi=220);plt.close(fig)

def real_plot(data_dir=None):
 if data_dir is None:return
 sys.path.insert(0,str(ROOT/"src"));from ssm_obs import mrclam,sim,ekf
 log=mrclam.load_robot(data_dir,1,.1,600);a=mrclam.audit_noise(log);ov,ow=a["odometry_robust_std"];mr,mb=a["measurement_robust_std"]
 p=sim.SensingParams(dt=.1,sig_v=max(ov,.01),sig_w=max(ow,.01),sig_r=max(mr,.03),sig_b=max(mb,.02),sensor_range=8,slip=1,gyro_bias=0,dropout_len=0,dropout_period=100000,n_landmarks=15)
 def run(o):
  o.reset(log["s0"]);return np.stack([o.step(log["Uodo"][k],log["Z"][k],log["M"][k],k) for k in range(len(log["S"]))])
 dr=run(ekf.DeadReckoning(p));e=run(ekf.EKF(p,log["landmarks"],q_scale=.25,r_scale=4))
 fig,ax=plt.subplots(figsize=(6.5,5.8));ax.plot(log["S"][:,0],log["S"][:,1],label="Vicon ground truth",linewidth=2);ax.plot(dr[:,0],dr[:,1],label="Dead reckoning");ax.plot(e[:,0],e[:,1],label="EKF");ax.scatter(log["landmarks"][:,0],log["landmarks"][:,1],marker="x",label="Landmarks");ax.set_xlabel("x (m)");ax.set_ylabel("y (m)");ax.set_title("UTIAS MRCLAM Robot 1: 600 s physical-log localization");ax.axis("equal");ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/"fig_mrclam_real_trajectory.png",dpi=220);plt.close(fig)

def main():
 proxy_plot();ablation_plot();seed_plot();controller_plot()
 data=os.environ.get("MRCLAM_DATA_DIR");real_plot(data)
 print("figures written to",OUT)
if __name__=="__main__":main()
