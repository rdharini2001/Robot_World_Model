"""Save / load trained observer parameters and their spec."""
import json, os
import numpy as np
from dataclasses import asdict
from .train import Spec


def save(path, spec, params):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path + ".npz", **params)
    with open(path + ".json", "w") as f:
        json.dump(asdict(spec), f, indent=2)


def load(path):
    with open(path + ".json") as f:
        spec = Spec(**json.load(f))
    d = np.load(path + ".npz")
    params = {k: d[k] for k in d.files}
    return spec, params
