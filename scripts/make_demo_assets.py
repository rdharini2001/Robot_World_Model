#!/usr/bin/env python3
"""Generate two narrated-demo-ready MP4 clips from fresh closed-loop simulations."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np

from ssm_obs import controllers, experiments as E, persist, sim

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo"
OUT.mkdir(exist_ok=True)

COLORS = {
    "dead_reckoning": "#80868b",
    "ekf": "#1a73e8",
    "gru_ekf": "#9334e6",
    "ssm_ekf": "#188038",
}
LABELS = {
    "dead_reckoning": "Dead reckoning",
    "ekf": "EKF",
    "gru_ekf": "GRU-EKF residual",
    "ssm_ekf": "Selective SSM-EKF residual",
}


def load_models():
    return {spec.name: persist.load(str(ROOT / "results/models" / spec.name)) for spec in E.ZOO}


def simulate(params, seed=3):
    trained = load_models()
    ref = sim.figure_eight(E.N_STEPS, params.dt, A=E.A_PATH)
    names = list(COLORS)
    runs = {}
    sensing = None
    for name in names:
        rng = np.random.default_rng(seed)
        active = sim.active_mask(params.n_landmarks, rng, randomize=True)
        observer = E.build(name, trained, params)
        controller = controllers.PurePursuit(ref[0], v_cmd=1.0, Ld=0.8)
        out = sim.rollout(controller, observer, params, sim.FIXED_MAP, ref, rng, active=active)
        runs[name] = out
        sensing = out["on"]
    return ref, runs, sensing


def render_clip(filename: str, params, title: str, seed: int):
    ref, runs, sensing = simulate(params, seed)
    path = ref[0]
    landmarks = sim.FIXED_MAP
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1])
    ax_traj = fig.add_subplot(grid[0, 0])
    ax_err = fig.add_subplot(grid[0, 1])

    ax_traj.plot(path[:, 0], path[:, 1], "--", color="black", lw=2, label="Reference")
    ax_traj.scatter(landmarks[:, 0], landmarks[:, 1], marker="*", s=130, color="#f9ab00", label="Landmarks")
    ax_traj.set_aspect("equal")
    ax_traj.set_xlim(path[:, 0].min() - 1.0, path[:, 0].max() + 1.0)
    ax_traj.set_ylim(path[:, 1].min() - 1.0, path[:, 1].max() + 1.0)
    ax_traj.set_xlabel("x position (m)")
    ax_traj.set_ylabel("y position (m)")
    ax_traj.set_title("Closed-loop trajectory")

    traj_lines = {}
    traj_dots = {}
    error_lines = {}
    errors = {}
    time = np.arange(E.N_STEPS) * params.dt
    for name, out in runs.items():
        traj_lines[name], = ax_traj.plot([], [], lw=2, color=COLORS[name], label=LABELS[name])
        traj_dots[name], = ax_traj.plot([], [], "o", ms=7, color=COLORS[name])
        errors[name] = np.linalg.norm(out["S"][:, :2] - out["E"][:, :2], axis=1)
        error_lines[name], = ax_err.plot([], [], lw=2, color=COLORS[name], label=LABELS[name])

    ax_err.set_xlim(0, time[-1])
    ymax = max(float(np.max(v)) for v in errors.values()) * 1.08
    ax_err.set_ylim(0, max(0.5, ymax))
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("position-estimation error (m)")
    ax_err.set_title("Estimator error seen by the controller")
    ax_err.grid(alpha=0.25)
    ax_traj.legend(loc="upper right", fontsize=8)
    ax_err.legend(loc="upper left", fontsize=8)

    status = fig.text(0.5, 0.94, "", ha="center", va="center", fontsize=16, weight="bold")
    fig.suptitle(title, fontsize=18, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(frame):
        k = frame + 1
        for name, out in runs.items():
            states = out["S"]
            traj_lines[name].set_data(states[:k, 0], states[:k, 1])
            traj_dots[name].set_data([states[k - 1, 0]], [states[k - 1, 1]])
            error_lines[name].set_data(time[:k], errors[name][:k])
        if bool(sensing[k - 1]):
            status.set_text("LANDMARK MEASUREMENTS AVAILABLE")
            status.set_color("#188038")
        else:
            status.set_text("SENSOR BLACKOUT - ODOMETRY ONLY")
            status.set_color("#d93025")
        return [*traj_lines.values(), *traj_dots.values(), *error_lines.values(), status]

    frames = list(range(0, E.N_STEPS, 2))
    animation = FuncAnimation(fig, update, frames=frames, interval=66, blit=True)
    writer = FFMpegWriter(fps=15, bitrate=3000, metadata={"title": title})
    out_path = OUT / filename
    animation.save(out_path, writer=writer)
    plt.close(fig)
    print("wrote", out_path.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["nominal", "blackout", "all"], default="all")
    args = parser.parse_args()
    if args.only in ("nominal", "all"):
        render_clip("nominal_tracking.mp4", E.nominal_eval(), "Nominal intermittent-sensing experiment", seed=3)
    if args.only in ("blackout", "all"):
        difficult = replace(E.nominal_eval(), dropout_len=30, dropout_period=42, gyro_bias=0.15, sig_r=0.25)
        render_clip("blackout_tracking.mp4", difficult, "Long-blackout and odometry-bias experiment", seed=7)


if __name__ == "__main__":
    main()
