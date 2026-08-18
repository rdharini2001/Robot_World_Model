"""Publication figures. Matplotlib only. One function per figure."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    "font.family": "DejaVu Sans",
})

COL = {
    "dead_reckoning": "#9aa0a6", "ekf": "#1a73e8",
    "gru_dr": "#e8710a", "ssm_dr": "#d93025",
    "ssm_ekf": "#137333", "gru_ekf": "#a142f4",
    "ssm_dr_nosel": "#c5221f", "ssm_dr_nomask": "#f29900",
}
MARK = {"dead_reckoning":"o","ekf":"s","gru_dr":"^","ssm_dr":"D","ssm_ekf":"*",
        "gru_ekf":"v","ssm_dr_nosel":"x","ssm_dr_nomask":"P"}
PRETTY = {
    "dead_reckoning":"Dead reckoning","ekf":"EKF","gru_dr":"GRU (DR-res.)",
    "ssm_dr":"Sel. SSM (DR-res.)","ssm_ekf":"Sel. SSM (EKF-res.)",
    "gru_ekf":"GRU (EKF-res.)","ssm_dr_nosel":"SSM no-selectivity",
    "ssm_dr_nomask":"SSM no-mask",
}
AXLABEL = {
    "dropout_len":"blackout duration (steps)", "sig_r":"range noise σ_r (m)",
    "n_landmarks":"# active landmarks", "gyro_bias":"gyro bias (rad/s)",
}


def _load(res, fn):
    with open(os.path.join(res, fn)) as f: return json.load(f)


def fig_sweeps(res, out, core, metric="ct_rmse", ylab="closed-loop cross-track RMSE (m)"):
    axes_order = ["dropout_len","sig_r","n_landmarks","gyro_bias"]
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 7.0))
    for ax, axis in zip(axs.flat, axes_order):
        sw = _load(res, f"sweep_{axis}.json"); lv = sw["_levels"]
        for n in core:
            m = np.array([r[metric][0] for r in sw[n]])
            e = np.array([r[metric][1] for r in sw[n]])
            ax.plot(lv, m, marker=MARK[n], color=COL[n], lw=1.8, ms=6, label=PRETTY[n])
            ax.fill_between(lv, m-e, m+e, color=COL[n], alpha=0.15, lw=0)
        ax.set_xlabel(AXLABEL[axis]); ax.set_ylabel(ylab)
        ax.set_title(axis.replace("_"," "), fontsize=10, loc="left", color="#444")
    axs.flat[0].legend(fontsize=8, ncol=1, loc="upper left")
    fig.suptitle("Closed-loop tracking degradation across four sensing axes", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(os.path.join(out, "fig_sweeps.png")); plt.close(fig)


def fig_mismatch(res, out):
    om = _load(res, "objective_mismatch.json")
    ol = np.array(om["scatter_ol"]); cl = np.array(om["scatter_cl"])
    labs = om["scatter_labels"]; core = om["core"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.3))
    for n in core:
        idx = [i for i,l in enumerate(labs) if l==n]
        a.scatter(ol[idx], cl[idx], c=COL[n], marker=MARK[n], s=42,
                  edgecolor="white", lw=0.4, label=PRETTY[n], alpha=0.9)
    a.set_xlabel("open-loop position RMSE (m)")
    a.set_ylabel("closed-loop cross-track RMSE (m)")
    a.set_title(f"Open-loop error vs. control cost\nSpearman ρ={om['spearman_global'][0]:.2f}, "
                f"Pearson r={om['pearson_global']:.2f}", fontsize=10, loc="left")
    a.legend(fontsize=7.5, loc="lower right")
    a.set_xscale("log"); a.set_yscale("log")
    # per-condition winner-flip bar
    rows = om["rank_rows"]
    flips = [1 if np.argmin(r["ol_rank"])!=np.argmin(r["cl_rank"]) else 0 for r in rows]
    rhos = [r["rho"] for r in rows]
    order = np.argsort(rhos)
    b.bar(range(len(rows)), np.array(rhos)[order],
          color=["#d93025" if flips[i] else "#1a73e8" for i in order], width=0.9)
    b.axhline(1.0, color="#888", lw=0.8, ls="--")
    b.set_xlabel("condition (sorted by rank agreement)")
    b.set_ylabel("per-condition Spearman ρ")
    b.set_title(f"Ranking agreement per condition\nwinner flips in "
                f"{int(sum(flips))}/{len(flips)} ({100*np.mean(flips):.0f}%) conditions",
                fontsize=10, loc="left")
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(color="#1a73e8",label="same winner"),
                      Patch(color="#d93025",label="winner flips")], fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_mismatch.png")); plt.close(fig)


def fig_timeseries(ts, out):
    on = np.array(ts["on"]); dt = ts["dt"]; t = np.arange(len(on))*dt
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    # shade blackout intervals
    off = ~on.astype(bool); k = 0; n = len(on); first=True
    while k < n:
        if off[k]:
            j = k
            while j < n and off[j]: j += 1
            ax.axvspan(t[k], t[min(j, n-1)], color="#ffd8d8", lw=0,
                       label="sensor blackout" if first else None); first=False
            k = j
        else: k += 1
    for n_ in ts["series"]:
        err = np.array(ts["series"][n_])
        ax.plot(t, err, color=COL[n_], lw=1.6, label=PRETTY[n_])
    ax.set_xlabel("time (s)"); ax.set_ylabel("position error (m)")
    ax.set_title("Estimation error through intermittent sensor blackouts", loc="left", fontsize=11)
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_timeseries.png")); plt.close(fig)


def fig_trajectory(trj, out):
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    P = np.array(trj["P"]); lm = np.array(trj["landmarks"])
    ax.plot(P[:,0], P[:,1], color="#222", lw=2.2, ls="--", label="reference", zorder=3)
    ax.scatter(lm[:,0], lm[:,1], marker="*", s=150, c="#f9ab00",
               edgecolor="#7a5900", lw=0.6, label="landmarks", zorder=4)
    for n_ in trj["traj"]:
        S = np.array(trj["traj"][n_])
        ax.plot(S[:,0], S[:,1], color=COL[n_], lw=1.6, alpha=0.9, label=PRETTY[n_])
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Closed-loop trajectories (estimator drives the controller)", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_trajectory.png")); plt.close(fig)


def fig_ablation(res, out):
    nom = _load(res, "nominal.json")
    groups = [("Prior", ["dead_reckoning","ekf"]),
              ("DR-residual", ["gru_dr","ssm_dr","ssm_dr_nosel","ssm_dr_nomask"]),
              ("EKF-residual", ["ekf","gru_ekf","ssm_ekf"])]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    x = 0; xt=[]; xl=[]
    for gname, names in groups:
        for n in names:
            v, e = nom[n]["ct_rmse"]
            ax.bar(x, v, yerr=e, color=COL.get(n,"#888"), width=0.8,
                   error_kw=dict(lw=1, capsize=3))
            ax.text(x, v+e+0.03, f"{v:.2f}", ha="center", fontsize=7.5)
            xt.append(x); xl.append(PRETTY[n]); x += 1
        x += 0.7
    ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("closed-loop cross-track RMSE (m)")
    ax.set_title("Nominal closed-loop tracking: priors, residual cells, and ablations",
                 loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_ablation.png")); plt.close(fig)
