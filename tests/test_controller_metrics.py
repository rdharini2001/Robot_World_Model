import numpy as np
from ssm_obs import controller_metrics as cm, controllers, sim


def test_zero_error_metrics_are_zero():
    ref = sim.figure_eight(60, 0.1, A=2.0)
    S = np.column_stack([ref[0], ref[1]])
    factory = lambda: controllers.TimeIndexedPurePursuit(ref[0], 1.0, 0.8)
    assert cm.controller_disagreement(S, S.copy(), factory) < 1e-12
    assert cm.local_control_sensitivity_error(S, S.copy(), factory) < 1e-12


def test_heading_error_changes_control_metric():
    ref = sim.figure_eight(60, 0.1, A=2.0)
    S = np.column_stack([ref[0], ref[1]])
    E = S.copy(); E[:, 2] += 0.2
    factory = lambda: controllers.TimeIndexedPurePursuit(ref[0], 1.0, 0.8)
    assert cm.controller_disagreement(S, E, factory) > 0
    assert cm.local_control_sensitivity_error(S, E, factory) > 0


def test_selection_summary_regret():
    proxy={"c":{"a":0.0,"b":1.0}}; closed={"c":{"a":2.0,"b":1.0}}
    out=cm.selection_summary(["c"],["a","b"],proxy,closed)
    assert out["flip_fraction"] == 1.0
    assert out["max_regret"] == 1.0
