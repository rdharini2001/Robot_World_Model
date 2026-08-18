import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from scipy.stats import spearmanr, rankdata
from ssm_obs import experiments as E
from ssm_obs.experiments import CORE, AXES

RES = os.path.join(os.path.dirname(__file__), "..", "results")

def load(fn):
    with open(os.path.join(RES, fn)) as f: return json.load(f)

def main():
    # Collect (open-loop pos RMSE, closed-loop ct RMSE) for CORE observers across
    # every (axis, level) condition. Each condition gives a 5-vector ranking.
    points_ol, points_cl = [], []      # flat scatter (per observer per condition)
    labels = []
    rank_rows = []                     # per-condition rank vectors
    for axis in AXES:
        sw = load(f"sweep_{axis}.json")
        levels = sw["_levels"]
        for i, val in enumerate(levels):
            ol = np.array([sw[n][i]["ol_pos_rmse"][0] for n in CORE])
            cl = np.array([sw[n][i]["ct_rmse"][0] for n in CORE])
            for j, n in enumerate(CORE):
                points_ol.append(ol[j]); points_cl.append(cl[j]); labels.append(n)
            # per-condition Spearman between open- and closed-loop rankings
            rho, _ = spearmanr(ol, cl)
            rank_rows.append({"axis": axis, "level": val, "rho": float(rho),
                              "ol_rank": rankdata(ol).tolist(),
                              "cl_rank": rankdata(cl).tolist()})
    ol = np.array(points_ol); cl = np.array(points_cl)
    rho_all, p_all = spearmanr(ol, cl)
    pear = np.corrcoef(ol, cl)[0, 1]
    # fraction of conditions where the open-loop winner != closed-loop winner
    flips = 0; tot = 0
    for r in rank_rows:
        olr, clr = np.array(r["ol_rank"]), np.array(r["cl_rank"])
        if np.argmin(olr) != np.argmin(clr): flips += 1
        tot += 1
    per_cond_rho = np.array([r["rho"] for r in rank_rows])
    out = {
        "n_points": len(ol),
        "spearman_global": [float(rho_all), float(p_all)],
        "pearson_global": float(pear),
        "per_condition_rho_mean": float(per_cond_rho.mean()),
        "per_condition_rho_min": float(per_cond_rho.min()),
        "winner_flip_fraction": flips / tot,
        "scatter_ol": ol.tolist(), "scatter_cl": cl.tolist(), "scatter_labels": labels,
        "rank_rows": rank_rows,
        "core": CORE,
    }
    with open(os.path.join(RES, "objective_mismatch.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"global Spearman(open-loop RMSE, closed-loop ct) = {rho_all:.3f} (p={p_all:.1e})")
    print(f"global Pearson = {pear:.3f}")
    print(f"per-condition rho: mean={per_cond_rho.mean():.3f} min={per_cond_rho.min():.3f}")
    print(f"winner-flip fraction = {flips}/{tot} = {flips/tot:.2f}")

if __name__ == "__main__":
    main()
