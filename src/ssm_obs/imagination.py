"""Multi-step "imagined rollout" diagnostics.

Motivation
----------
Every observer in this project is really a *minimal world model*: a
recursive function that consumes actions (odometry) and, when available,
observations, and predicts the agent's future state. The paper's central
open-loop-vs-closed-loop story (replay RMSE vs cross-track RMSE) only ever
tests *single-step, measurement-corrected* prediction: at every timestep the
observer is handed the true logged measurement (or told none arrived) and
scored on that one step.

Large-scale video/latent world models (Genie, Cosmos, Dreamer, ...) are
instead evaluated -- and, more importantly, *used* -- by imagining forward
for many steps under a candidate action sequence with **no** corrective
observation. This module adds that evaluation mode to the six-observer
testbed: given a logged trajectory (real odometry, real actions), we replay
it through a fresh observer, and at regular checkpoints we fork the
observer's internal state and let it "dream" forward for H steps using only
the realized control sequence -- exactly the action-conditioned,
observation-free rollout that a world model would be asked to produce for
planning or evaluation.

Because every observer in this codebase exposes the same tiny interface
(`reset(s0)`, `step(u_odo, z, mask, k) -> pose`) and only ever touches its
own plain-attribute state, forking is just `copy.deepcopy`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from . import dynamics as dyn


@dataclass(frozen=True)
class ImaginationSpec:
    """Configuration for a blind, action-conditioned rollout probe."""
    horizon: int = 20      # steps to imagine forward with no observation
    stride: int = 10       # spacing between fork points along the trajectory
    warmup: int = 10       # skip the first `warmup` steps (let the filter settle)


def _blank_measurement(n_landmarks: int):
    return np.zeros((n_landmarks, 2)), np.zeros(n_landmarks)


def imagine_from_fork(estimator, u_odo_future: np.ndarray, n_landmarks: int) -> np.ndarray:
    """Fork `estimator` (already deep-copied by the caller) and roll it
    forward blind (no measurement) for len(u_odo_future) steps.

    Returns the imagined pose sequence, shape (H, 3).
    """
    z0, m0 = _blank_measurement(n_landmarks)
    H = len(u_odo_future)
    poses = np.zeros((H, 3))
    for h in range(H):
        poses[h] = estimator.step(u_odo_future[h], z0, m0, k=-1)
    return poses


def imagination_rollout(estimator, out: Dict, spec: ImaginationSpec = ImaginationSpec()) -> Dict:
    """Replay a logged trajectory through `estimator`, forking at `stride`
    intervals to probe `horizon`-step blind (imagined) prediction.

    `out` is any logged rollout dict produced by `sim.rollout` /
    `sim.rollout_openloop` (needs S, Uodo, Z, M, and a start pose S[0]).
    `estimator` must be freshly constructed (unreset); this function calls
    `.reset(...)` itself so the *real* (measurement-corrected) trajectory
    reproduced here is deterministic given the logged inputs.

    Returns a dict with:
      errors        : (n_forks, horizon) position error at each imagined step
      final_errors  : (n_forks,) position error at exactly horizon H
      heading_errors: (n_forks, horizon) heading error at each imagined step
      fork_steps    : (n_forks,) the timestep each fork happened at
    """
    S, Uodo, Z, M = out["S"], out["Uodo"], out["Z"], out["M"]
    n = len(S)
    n_lm = Z.shape[1]
    estimator.reset(S[0].copy())

    fork_steps: List[int] = []
    errs: List[np.ndarray] = []
    herrs: List[np.ndarray] = []

    for k in range(n):
        # Real (corrected) step first, so internal state tracks ground truth.
        estimator.step(Uodo[k], Z[k], M[k], k)

        due = (k >= spec.warmup) and ((k - spec.warmup) % spec.stride == 0)
        have_future = k + spec.horizon < n
        if due and have_future:
            fork = copy.deepcopy(estimator)
            u_future = Uodo[k + 1: k + 1 + spec.horizon]
            imagined = imagine_from_fork(fork, u_future, n_lm)
            true_future = S[k + 1: k + 1 + spec.horizon]
            dpos = np.linalg.norm(imagined[:, :2] - true_future[:, :2], axis=1)
            dhead = np.abs(dyn.wrap(imagined[:, 2] - true_future[:, 2]))
            fork_steps.append(k)
            errs.append(dpos)
            herrs.append(dhead)

    if not errs:
        return dict(errors=np.zeros((0, spec.horizon)), final_errors=np.zeros(0),
                    heading_errors=np.zeros((0, spec.horizon)), fork_steps=np.zeros(0, dtype=int))

    E = np.stack(errs)          # (n_forks, horizon)
    HE = np.stack(herrs)
    return dict(errors=E, final_errors=E[:, -1].copy(), heading_errors=HE,
                fork_steps=np.array(fork_steps, dtype=int))


def imagination_summary(result: Dict) -> Dict[str, float]:
    """Scalar summary of an `imagination_rollout` result.

    `imagine_rmse`      : RMSE of imagined position error at the FULL horizon H
                           (the headline "does the world model's H-step dream
                           match reality" number).
    `imagine_auc`       : mean position error averaged over all imagined
                           horizons 1..H (integrates the whole growth curve,
                           not just the endpoint).
    `imagine_head_rmse` : heading analogue of `imagine_rmse`.
    """
    E, HE = result["errors"], result["heading_errors"]
    if E.size == 0:
        return dict(imagine_rmse=float("nan"), imagine_auc=float("nan"),
                    imagine_head_rmse=float("nan"))
    final = E[:, -1]
    return dict(
        imagine_rmse=float(np.sqrt(np.mean(final ** 2))),
        imagine_auc=float(np.mean(E)),
        imagine_head_rmse=float(np.sqrt(np.mean(HE[:, -1] ** 2))),
    )


def growth_curve(result: Dict) -> np.ndarray:
    """Mean position error as a function of imagined horizon step, shape (H,).
    This is the "how fast does the dream diverge from reality" curve used
    for the rollout-consistency figure.
    """
    E = result["errors"]
    if E.size == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(E ** 2, axis=0))
