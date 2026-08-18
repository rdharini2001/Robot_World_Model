"""UTIAS MRCLAM loading, noise auditing, and empirical-noise replay utilities.

Raw UTIAS files are intentionally not redistributed. Use
``scripts/download_mrclam.py`` or point ``--data-dir`` to an existing dataset.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from . import dynamics as dyn, sim


def _load(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.loadtxt(path, comments="#")


def _interp_angle(t_new, t, theta):
    return dyn.wrap(np.interp(t_new, t, np.unwrap(theta)))


def load_robot(data_dir, robot=1, dt=0.1, duration=None, max_landmarks=None):
    """Load one robot and align GT, odometry, and range-bearing data to a grid.

    Returns a simulator-compatible log dictionary. Measurements identify
    landmarks through ``Barcodes.dat``; observations of other robots are
    excluded.
    """
    data_dir = Path(data_dir)
    bar = _load(data_dir / "Barcodes.dat")
    barcode_to_subject = {int(row[1]): int(row[0]) for row in bar}
    lm_raw = _load(data_dir / "Landmark_Groundtruth.dat")
    subjects = lm_raw[:, 0].astype(int)
    order = np.argsort(subjects); lm_raw = lm_raw[order]; subjects = subjects[order]
    if max_landmarks is not None:
        lm_raw = lm_raw[:max_landmarks]; subjects = subjects[:max_landmarks]
    landmarks = lm_raw[:, 1:3]
    subject_to_idx = {int(s): i for i, s in enumerate(subjects)}

    gt = _load(data_dir / f"Robot{robot}_Groundtruth.dat")
    odo = _load(data_dir / f"Robot{robot}_Odometry.dat")
    meas = _load(data_dir / f"Robot{robot}_Measurement.dat")
    # Keep landmark observations only (subjects 6+ in the official data).
    subject = np.array([barcode_to_subject.get(int(b), -1) for b in meas[:, 1]], dtype=int)
    keep = np.array([s in subject_to_idx for s in subject])
    meas = meas[keep]; subject = subject[keep]

    t0 = max(gt[0, 0], odo[0, 0], meas[0, 0])
    t1 = min(gt[-1, 0], odo[-1, 0], meas[-1, 0])
    if duration is not None:
        t1 = min(t1, t0 + float(duration))
    t = np.arange(t0, t1, dt)
    S = np.column_stack([
        np.interp(t, gt[:, 0], gt[:, 1]),
        np.interp(t, gt[:, 0], gt[:, 2]),
        _interp_angle(t, gt[:, 0], gt[:, 3]),
    ])
    Uodo = np.column_stack([
        np.interp(t, odo[:, 0], odo[:, 1]),
        np.interp(t, odo[:, 0], odo[:, 2]),
    ])
    K = len(landmarks)
    Z = np.zeros((len(t), K, 2)); M = np.zeros((len(t), K))
    # Assign each timestamp to nearest grid point and average duplicate readings.
    sums = np.zeros_like(Z); counts = np.zeros_like(M)
    idx = np.rint((meas[:, 0] - t0) / dt).astype(int)
    valid = (idx >= 0) & (idx < len(t))
    for row, subj, k in zip(meas[valid], subject[valid], idx[valid]):
        j = subject_to_idx[int(subj)]
        sums[k, j, 0] += row[2]
        sums[k, j, 1] += row[3]
        counts[k, j] += 1
    nz = counts > 0
    Z[..., 0][nz] = sums[..., 0][nz] / counts[nz]
    # Circular averaging is unnecessary for duplicates at the same timestamp,
    # but wrapping keeps a valid bearing convention.
    Z[..., 1][nz] = dyn.wrap(sums[..., 1][nz] / counts[nz])
    M[nz] = 1.0
    return {
        "t": t - t[0], "S": S, "Uodo": Uodo, "Z": Z, "M": M,
        "on": M.sum(axis=1) > 0, "landmarks": landmarks,
        "subjects": subjects, "s0": S[0].copy(), "dt": float(dt),
        "source": f"UTIAS MRCLAM robot {robot}",
    }


def groundtruth_controls(log):
    """Estimate body-frame v and yaw rate from motion-capture ground truth."""
    S=np.asarray(log["S"]); dt=float(log["dt"]); th=np.unwrap(S[:,2])
    dx=np.gradient(S[:,0],dt); dy=np.gradient(S[:,1],dt)
    v=dx*np.cos(th)+dy*np.sin(th); w=np.gradient(th,dt)
    return np.column_stack([v,w])


def measurement_residuals(log):
    rows=[]
    for k,s in enumerate(log["S"]):
        for j,lm in enumerate(log["landmarks"]):
            if log["M"][k,j] < .5: continue
            pred=dyn.meas_predict(s,lm)
            e=log["Z"][k,j]-pred; e[1]=dyn.wrap(e[1]); rows.append(e)
    return np.asarray(rows,dtype=float)


def robust_scale(x):
    x=np.asarray(x); med=np.median(x,axis=0); mad=np.median(np.abs(x-med),axis=0)
    return med,1.4826*mad+1e-8


def audit_noise(log):
    Utrue=groundtruth_controls(log); odo_res=log["Uodo"]-Utrue
    meas_res=measurement_residuals(log)
    odo_bias,odo_scale=robust_scale(odo_res); meas_bias,meas_scale=robust_scale(meas_res)
    counts=log["M"].sum(axis=1)
    return {
        "n_steps":int(len(log["S"])),"duration_s":float(log["t"][-1]),
        "n_landmarks":int(len(log["landmarks"])),"n_measurements":int(log["M"].sum()),
        "measurement_step_fraction":float(np.mean(counts>0)),
        "measurements_per_active_step":float(np.mean(counts[counts>0])) if np.any(counts>0) else 0.0,
        "odometry_bias":odo_bias.tolist(),"odometry_robust_std":odo_scale.tolist(),
        "measurement_bias":meas_bias.tolist(),"measurement_robust_std":meas_scale.tolist(),
    }


@dataclass
class EmpiricalNoise:
    odometry_residuals: np.ndarray
    measurement_residuals: np.ndarray
    measurement_counts: np.ndarray


def fit_empirical_noise(log, center=False):
    odo=log["Uodo"]-groundtruth_controls(log)
    meas=measurement_residuals(log)
    if center:
        odo=odo-np.median(odo,axis=0); meas=meas-np.median(meas,axis=0)
    return EmpiricalNoise(odo,meas,log["M"].sum(axis=1).astype(int))


def rollout_empirical(controller, estimator, params, landmarks, ref, noise, rng, start=None):
    """Closed-loop simulation driven by residual/count samples from MRCLAM.

    True dynamics remain simulated. Odometry error, range-bearing residuals,
    and the number/timing of observations are resampled from physical logs.
    This is therefore a real-noise calibrated replay, not a hardware run.
    """
    P,th_ref,v_ref,w_ref=ref; n=len(P); dt=params.dt; K=len(landmarks)
    s=np.array([P[0,0],P[0,1],th_ref[0]]) if start is None else np.array(start,float)
    estimator.reset(s.copy())
    S=np.zeros((n,3)); E=np.zeros((n,3)); U=np.zeros((n,2)); Uodo=np.zeros((n,2))
    Z=np.zeros((n,K,2)); M=np.zeros((n,K)); on=np.zeros(n,dtype=bool)
    # Use a contiguous real count sequence so blackout autocorrelation is kept.
    counts=noise.measurement_counts
    start_i=int(rng.integers(0,max(1,len(counts)-n)))
    seq=np.resize(counts[start_i:start_i+n],n)
    est_pose=s.copy()
    for k in range(n):
        u=np.clip(controller(est_pose,k),[0.0,-3.0],[2.5,3.0])
        oe=noise.odometry_residuals[rng.integers(len(noise.odometry_residuals))]
        u_odo=u+oe
        # Select the closest visible landmarks, capped by the empirical count.
        dist=np.linalg.norm(landmarks-s[:2],axis=1)
        candidates=np.where(dist <= params.sensor_range)[0]
        c=min(int(seq[k]),len(candidates))
        if c>0:
            # Randomize within distance-biased candidates to avoid deterministic slots.
            weights=np.exp(-dist[candidates]/max(params.sensor_range,1e-6)); weights/=weights.sum()
            chosen=rng.choice(candidates,size=c,replace=False,p=weights)
            for j in chosen:
                e=noise.measurement_residuals[rng.integers(len(noise.measurement_residuals))]
                Z[k,j]=dyn.meas_predict(s,landmarks[j])+e; Z[k,j,1]=dyn.wrap(Z[k,j,1]); M[k,j]=1
            on[k]=True
        est_pose=estimator.step(u_odo,Z[k],M[k],k)
        S[k]=s;E[k]=est_pose;U[k]=u;Uodo[k]=u_odo
        s=dyn.step_true(s,u,dt)
    return dict(S=S,E=E,U=U,Uodo=Uodo,Z=Z,M=M,on=on,P=P,th_ref=th_ref,
                v_ref=v_ref,w_ref=w_ref,landmarks=landmarks)
