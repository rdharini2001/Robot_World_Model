import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from ssm_obs import persist, sim, controllers
from ssm_obs import experiments as E
from ssm_obs.experiments import ZOO, N_STEPS, A_PATH

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
trained = {s.name: persist.load(os.path.join(RES, "models", s.name)) for s in ZOO}

ps = E.nominal_eval()
ref = sim.figure_eight(N_STEPS, ps.dt, A=A_PATH)
names = ["dead_reckoning", "ekf", "ssm_ekf"]
col = {"dead_reckoning": "#9aa0a6", "ekf": "#1a73e8", "ssm_ekf": "#137333"}
pretty = {"dead_reckoning": "Dead reckoning", "ekf": "EKF", "ssm_ekf": "Sel. SSM (EKF-res.)"}
runs = {}; on_sched = None
for n in names:
    rng = np.random.default_rng(3)
    active = sim.active_mask(ps.n_landmarks, rng, randomize=True)
    est = E.build(n, trained, ps)
    ctrl = controllers.PurePursuit(ref[0], v_cmd=1.0, Ld=0.8)
    out = sim.rollout(ctrl, est, ps, sim.FIXED_MAP, ref, rng, active=active)
    runs[n] = out["S"][:, :2]; on_sched = out["on"]

P = ref[0]; lm = sim.FIXED_MAP
fig, ax = plt.subplots(figsize=(6.2, 6.0))
ax.plot(P[:, 0], P[:, 1], "--", color="#222", lw=1.8, label="reference")
ax.scatter(lm[:, 0], lm[:, 1], marker="*", s=140, c="#f9ab00", edgecolor="#7a5900", lw=0.6, label="landmarks")
lines = {n: ax.plot([], [], color=col[n], lw=2.0, label=pretty[n])[0] for n in names}
dots = {n: ax.plot([], [], "o", color=col[n], ms=7)[0] for n in names}
ax.set_aspect("equal"); ax.set_xlim(P[:,0].min()-1, P[:,0].max()+1); ax.set_ylim(P[:,1].min()-1, P[:,1].max()+1)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
banner = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=11, weight="bold")

STEP = 2
frames = list(range(1, N_STEPS, STEP))
def upd(k):
    for n in names:
        S = runs[n]
        lines[n].set_data(S[:k, 0], S[:k, 1]); dots[n].set_data([S[k-1, 0]], [S[k-1, 1]])
    if not on_sched[k-1]:
        banner.set_text("SENSOR BLACKOUT"); banner.set_color("#d93025")
    else:
        banner.set_text("sensing"); banner.set_color("#137333")
    return list(lines.values()) + list(dots.values()) + [banner]

anim = FuncAnimation(fig, upd, frames=frames, blit=True)
out = os.path.join(FIG, "demo_trajectory.gif")
anim.save(out, writer=PillowWriter(fps=15))
print("wrote", out)
