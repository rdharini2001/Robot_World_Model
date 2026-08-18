import argparse
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssm_obs import data, persist
from ssm_obs.experiments import ZOO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*", help="Optional model names")
    p.add_argument("--backend", choices=["auto", "torch", "autograd"], default="auto")
    p.add_argument("--n-train", type=int, default=int(os.environ.get("N_TRAIN", 200)))
    p.add_argument("--n-steps", type=int, default=int(os.environ.get("N_STEPS", 200)))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def select_backend(name):
    if name in ("auto", "autograd"):
        try:
            from ssm_obs import train as backend
            if backend.grad is not None:
                return backend, "autograd"
            if name == "autograd":
                raise RuntimeError("autograd is unavailable")
        except Exception:
            if name == "autograd":
                raise
    from ssm_obs import torch_train as backend
    return backend, "torch"


def main():
    args = parse_args(); selected = set(args.models)
    backend, backend_name = select_backend(args.backend)
    modeldir = os.path.join(os.path.dirname(__file__), "..", "results", "models")
    print(f"backend={backend_name}; generating {args.n_train} x {args.n_steps} training data")
    t0 = time.time()
    ds = data.generate(args.n_train, args.n_steps, seed=1)
    val = data.generate(max(16, args.n_train // 12), args.n_steps, seed=2)
    print(f"data ready in {time.time()-t0:.1f}s")
    for spec in ZOO:
        if selected and spec.name not in selected:
            continue
        path = os.path.join(modeldir, spec.name)
        if os.path.exists(path + ".npz") and not args.force:
            print(f"[skip] {spec.name}")
            continue
        print(f"[train] {spec.name}")
        t0 = time.time(); params = backend.train(spec, ds, val, verbose=True)
        persist.save(path, spec, params)
        print(f"saved {spec.name} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
