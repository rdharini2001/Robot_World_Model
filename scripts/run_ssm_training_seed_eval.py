#!/usr/bin/env python3
from __future__ import annotations
import json,os,sys
from pathlib import Path
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","src"))
import numpy as np
from ssm_obs import experiments as E,persist
ROOT=Path(__file__).resolve().parents[1]
PATHS={0:ROOT/"results"/"models"/"ssm_ekf",1:ROOT/"results"/"models_multiseed"/"ssm_ekf_seed1",2:ROOT/"results"/"models_multiseed"/"ssm_ekf_seed2"}
CONDS={"nominal":E.nominal_eval(),"long_blackout":E._apply(E.nominal_eval(),"dropout_len",34),"high_range_noise":E._apply(E.nominal_eval(),"sig_r",.5),"two_landmarks":E._apply(E.nominal_eval(),"n_landmarks",2),"high_gyro_bias":E._apply(E.nominal_eval(),"gyro_bias",.25)}
def main():
 out={"model":"ssm_ekf","training_seeds":[0,1,2],"conditions":{}}
 for label,p in CONDS.items():
  print(label,flush=True);out["conditions"][label]={};means=[]
  for sd,path in PATHS.items():
   spec,w=persist.load(str(path));trained={"ssm_ekf":(spec,w)}
   vals=E.eval_closed("ssm_ekf",trained,p,list(range(10)));x=[v["ct_rmse"] for v in vals]
   m=float(np.mean(x));means.append(m);out["conditions"][label][str(sd)]={"mean":m,"ci95_half":float(1.96*np.std(x,ddof=1)/np.sqrt(len(x))),"values":x}
  out["conditions"][label]["across_training_seeds"]={"mean":float(np.mean(means)),"std":float(np.std(means,ddof=1)),"min":float(np.min(means)),"max":float(np.max(means)),"values":means}
  json.dump(out,open(ROOT/"results"/"enhanced"/"ssm_training_seed_eval.json","w"),indent=2)
 print(json.dumps(out["conditions"],indent=2))
if __name__=="__main__":main()
