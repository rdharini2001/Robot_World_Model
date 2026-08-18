import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from ssm_obs import persist, sim, metrics, controllers, data as data_mod
from ssm_obs import experiments as E
from ssm_obs.experiments import ZOO, CORE, AXES, N_STEPS, A_PATH

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
NSEED = int(os.environ.get("NSEED", 10))
SEEDS = list(range(NSEED))
ALL = CORE + ["ssm_dr_nosel", "ssm_dr_nomask", "ssm_ekf_nosel", "ssm_ekf_nomask"]

def load_zoo():
    trained = {}
    for spec in ZOO:
        s, w = persist.load(os.path.join(RES, "models", spec.name))
        trained[spec.name] = (s, w)
    return trained

def agg(ms, key):
    v = np.array([m[key] for m in ms]); return float(v.mean()), float(v.std()/max(1,len(v))**0.5*1.96)

def dump(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote results/{name}")

def do_nominal(trained):
    if os.path.exists(os.path.join(RES, "nominal.json")):
        print("[skip] nominal"); return
    ps = E.nominal_eval(); out = {}
    for name in ALL:
        cl = E.eval_closed(name, trained, ps, SEEDS)
        ol = E.eval_open(name, trained, ps, SEEDS)
        row = {k: agg(cl, k) for k in ["pos_rmse","head_rmse","ct_rmse","ct_max","effort","recovery","diverge"]}
        opr = np.array([r[0] for r in ol]); ohr = np.array([r[1] for r in ol])
        row["ol_pos_rmse"] = [float(opr.mean()), float(opr.std()/len(opr)**0.5*1.96)]
        row["ol_head_rmse"] = [float(ohr.mean()), float(ohr.std()/len(ohr)**0.5*1.96)]
        out[name] = row
        print(f"  {name:16s} clPos={row['pos_rmse'][0]:.3f} ct={row['ct_rmse'][0]:.3f} olPos={row['ol_pos_rmse'][0]:.3f}")
    dump("nominal.json", out)

def do_axis(axis, trained):
    fn = f"sweep_{axis}.json"
    if os.path.exists(os.path.join(RES, fn)):
        print(f"[skip] {axis}"); return
    base = E.nominal_eval(); out = {n: [] for n in ALL}; out["_levels"] = AXES[axis]
    for val in AXES[axis]:
        ps = E._apply(base, axis, val)
        for name in ALL:
            cl = E.eval_closed(name, trained, ps, SEEDS)
            ol = E.eval_open(name, trained, ps, SEEDS)
            rec = {k: agg(cl, k) for k in ["pos_rmse","ct_rmse","effort","recovery","diverge"]}
            opr = np.array([r[0] for r in ol])
            rec["ol_pos_rmse"] = [float(opr.mean()), float(opr.std()/len(opr)**0.5*1.96)]
            out[name].append(rec)
        print(f"  {axis}={val}: done")
    dump(fn, out)

def main():
    trained = load_zoo()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("all", "nominal"):
        print("[nominal]"); do_nominal(trained)
    for axis in AXES:
        if which in ("all", axis):
            print(f"[sweep {axis}]"); do_axis(axis, trained)
    print(f"done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
