import numpy as np
from pathlib import Path
from ssm_obs import mrclam


def test_mrclam_loader_shapes():
    data = Path('/mnt/data/mrclam1')
    if not data.exists():
        return
    log = mrclam.load_robot(data, robot=1, dt=0.2, duration=10)
    assert log['S'].shape[1] == 3
    assert log['Uodo'].shape[1] == 2
    assert log['Z'].shape[:2] == log['M'].shape
    assert log['Z'].shape[2] == 2
    assert len(log['landmarks']) == log['M'].shape[1]
    audit = mrclam.audit_noise(log)
    assert audit['n_steps'] == len(log['S'])
