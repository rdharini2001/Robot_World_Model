import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ssm_obs import sim, experiments as E, persist, imagination as im, data as data_mod

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
    "ssm_ekf_imagine_seed0": "#0b8043", "gru_ekf_imagine_seed0": "#7627bb",
    "ssm_dr_imagine_seed0": "#b31412",
}
MARK = {"dead_reckoning": "o", "ekf": "s", "gru_dr": "^", "ssm_dr": "D",
        "ssm_ekf": "*", "gru_ekf": "v",
        "ssm_ekf_imagine_seed0": "*", "gru_ekf_imagine_seed0": "v",
        "ssm_dr_imagine_seed0": "D"}
LS = {"ssm_ekf_imagine_seed0": "--", "gru_ekf_imagine_seed0": "--", "ssm_dr_imagine_seed0": "--"}
PRETTY = {
    "dead_reckoning": "Dead reckoning", "ekf": "EKF", "gru_dr": "GRU (DR-res.)",
    "ssm_dr": "Sel. SSM (DR-res.)", "ssm_ekf": "Sel. SSM (EKF-res.)",
    "gru_ekf": "GRU (EKF-res.)",
    "ssm_ekf_imagine_seed0": "Sel. SSM (EKF-res.), imagination-trained",
    "gru_ekf_imagine_seed0": "GRU (EKF-res.), imagination-trained",
    "ssm_dr_imagine_seed0": "Sel. SSM (DR-res.), imagination-trained",
}

RES = os.path.join(os.path.dirname(__file__), "..", "results", "worldmodel")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures", "worldmodel")
os.makedirs(FIG, exist_ok=True)


def _load(fn):
    with open(os.path.join(RES, fn)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
def fig_growth_curves():
    """Blind imagination error vs imagined horizon step, nominal condition."""
    trained = {}
    for spec in E.ZOO:
        s, w = persist.load(os.path.join("results", "models", spec.name))
        trained[spec.name] = (s, w)
    for name in ["ssm_ekf_imagine_seed0", "gru_ekf_imagine_seed0", "ssm_dr_imagine_seed0"]:
        s, w = persist.load(os.path.join("results", "models_worldmodel", name))
        trained[name] = (s, w)

    ps = E.nominal_eval()
    spec_im = im.ImaginationSpec(horizon=20, stride=10, warmup=10)
    names = ["dead_reckoning", "ekf", "gru_dr", "ssm_dr", "gru_ekf", "ssm_ekf",
             "gru_ekf_imagine_seed0", "ssm_ekf_imagine_seed0"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for name in names:
        curves = []
        for sd in range(10):
            rng = np.random.default_rng(5000 + sd)
            traj = data_mod.expert_rollout(ps, E.N_STEPS, rng)
            out = dict(S=traj["S"], Uodo=traj["Uodo"], Z=traj["Z"], M=traj["M"])
            est = E.build(name, trained, ps)
            res = im.imagination_rollout(est, out, spec_im)
            c = im.growth_curve(res)
            if len(c):
                curves.append(c)
        curve = np.mean(np.stack(curves), axis=0)
        ax = axL if name in ("dead_reckoning", "ekf", "gru_dr", "ssm_dr") else axR
        ax.plot(np.arange(1, len(curve) + 1) * ps.dt, curve,
                marker=MARK.get(name, "o"), color=COL[name], lw=1.8, ms=5,
                ls=LS.get(name, "-"), label=PRETTY[name])
    for ax, title in [(axL, "baselines"), (axR, "EKF-anchored: standard vs. imagination-trained")]:
        ax.set_xlabel("imagined horizon (s)")
        ax.set_ylabel("blind rollout position error (m)")
        ax.set_title(title, fontsize=10, loc="left", color="#444")
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("How fast does each observer's blind (measurement-free) forward "
                  "prediction diverge from reality?", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "fig_imagination_growth.png"))
    plt.close(fig)
    print("wrote fig_imagination_growth.png")


# ---------------------------------------------------------------------------
def fig_proxy_scatter():
    d = _load("proxy_with_imagination.json")
    raw = d["raw"]
    conds, closed, proxy_pose, proxy_im = raw["conditions"], raw["closed"], raw["proxy_pose"], raw["proxy_imagine"]
    core = E.CORE

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for ax, proxy, title, rho in [
        (axL, proxy_pose, "single-step replay position RMSE", d["position_rmse"]["spearman"]),
        (axR, proxy_im, "20-step blind imagination RMSE", d["imagine_rmse_H20"]["spearman"]),
    ]:
        for n in core:
            xs = [proxy[c][n] for c in conds]
            ys = [closed[c][n] for c in conds]
            ax.scatter(xs, ys, s=26, color=COL[n], marker=MARK[n], label=PRETTY[n], alpha=0.85)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"offline proxy: {title} (m)")
        ax.set_ylabel("closed-loop cross-track RMSE (m)")
        ax.set_title(f"ρ = {rho:.3f}", fontsize=10, loc="left", color="#444")
    axL.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Neither offline proxy is a perfect stand-in for closed-loop utility -- "
                 "the imagination-rollout proxy is the weaker of the two here", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "fig_proxy_scatter.png"))
    plt.close(fig)
    print("wrote fig_proxy_scatter.png")


# ---------------------------------------------------------------------------
def fig_selection_criteria_bar():
    d = _load("proxy_with_imagination.json")
    rows = [
        ("Position RMSE", "position_rmse", "#1a73e8"),
        ("Heading RMSE", "heading_rmse", "#e8710a"),
        ("Imagination RMSE (H=20)", "imagine_rmse_H20", "#137333"),
    ]
    labels = [r[0] for r in rows]
    flips = [d[r[1]]["flip_fraction"] * 24 for r in rows]
    maxregret = [d[r[1]]["max_regret"] for r in rows]
    colors = [r[2] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    ax1.bar(labels, flips, color=colors)
    ax1.set_ylabel("top-1 failures (of 24 conditions)")
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax2.bar(labels, maxregret, color=colors)
    ax2.set_ylabel("max regret (m)")
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    fig.suptitle("Offline selection criteria: position RMSE remains the strongest simple proxy",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG, "fig_selection_criteria_bar.png"))
    plt.close(fig)
    print("wrote fig_selection_criteria_bar.png")


# ---------------------------------------------------------------------------
def fig_dropout_curriculum():
    sw = _load("wm_sweep_dropout_len.json")
    levels = sw["_levels"]
    pairs = [("ssm_ekf", "ssm_ekf_imagine_seed0"), ("gru_ekf", "gru_ekf_imagine_seed0"),
             ("ssm_dr", "ssm_dr_imagine_seed0")]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=False)
    for ax, (std, imn) in zip(axs, pairs):
        for name in (std, imn):
            m = np.array([r["ct_rmse"][0] for r in sw[name]])
            e = np.array([r["ct_rmse"][1] for r in sw[name]])
            ax.plot(levels, m, marker=MARK.get(name, "o"), color=COL[name], lw=1.8, ms=6,
                    ls=LS.get(name, "-"), label=PRETTY[name])
            ax.fill_between(levels, m - e, m + e, color=COL[name], alpha=0.15, lw=0)
        ax.set_xlabel("blackout duration (steps)")
        ax.set_ylabel("closed-loop cross-track RMSE (m)")
        ax.legend(fontsize=7.2, loc="upper left")
    fig.suptitle("Imagination-curriculum training vs. standard training across the blackout sweep",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG, "fig_dropout_curriculum.png"))
    plt.close(fig)
    print("wrote fig_dropout_curriculum.png")


# ---------------------------------------------------------------------------
def fig_compound_bar():
    d = _load("wm_compound.json")
    pairs = [("ekf", None), ("ssm_ekf", "ssm_ekf_imagine_seed0"),
             ("gru_ekf", "gru_ekf_imagine_seed0"), ("ssm_dr", "ssm_dr_imagine_seed0")]
    labels, vals, colors = [], [], []
    for std, imn in pairs:
        labels.append(PRETTY.get(std, std)); vals.append(d[std]["ct_rmse"][0]); colors.append(COL[std])
        if imn:
            labels.append(PRETTY[imn]); vals.append(d[imn]["ct_rmse"][0]); colors.append(COL[imn])

    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8)
    ax.set_ylabel("closed-loop cross-track RMSE (m)")
    ax.set_title("Compound-stress condition: imagination-curriculum training helps the "
                 "EKF-anchored arms, not the DR-anchored one", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_compound_bar.png"))
    plt.close(fig)
    print("wrote fig_compound_bar.png")


if __name__ == "__main__":
    fig_growth_curves()
    fig_proxy_scatter()
    fig_selection_criteria_bar()
    fig_dropout_curriculum()
    fig_compound_bar()
