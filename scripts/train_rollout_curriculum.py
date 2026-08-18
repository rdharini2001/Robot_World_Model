"""Train the imagination-curriculum observer arm.

These models share the exact architecture and hyperparameters of the
matching entry in experiments.ZOO (ssm_ekf / ssm_dr / gru_ekf); the ONLY
difference is the training data: `data.generate_imagination_curriculum`
carves a synthetic long (32-58 step) blackout window into half of the
training trajectories, well beyond the naturally sampled dropout_len range
(6-27), so the network is explicitly exposed to -- and scored through --
extended blind-imagination windows during training. This isolates the
effect of the training curriculum from the effect of architecture, which the
existing matched-anchor ablation (run_matched_ablation.py) does not test.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ssm_obs import data, persist, torch_train
from ssm_obs.train import Spec

MODELDIR = os.path.join(os.path.dirname(__file__), "..", "results", "models_worldmodel")


def specs():
    return [
        Spec(name="ssm_ekf_imagine_seed0", cell="ssm", anchor="ekf", d=32, epochs=32,
             lr=3e-3, selective=True, use_cov=True, seed=0),
        Spec(name="ssm_ekf_imagine_seed1", cell="ssm", anchor="ekf", d=32, epochs=32,
             lr=3e-3, selective=True, use_cov=True, seed=1),
        Spec(name="ssm_ekf_imagine_seed2", cell="ssm", anchor="ekf", d=32, epochs=32,
             lr=3e-3, selective=True, use_cov=True, seed=2),
        Spec(name="ssm_dr_imagine_seed0", cell="ssm", anchor="dr", d=32, epochs=32,
             lr=4e-3, selective=True, seed=0),
        Spec(name="gru_ekf_imagine_seed0", cell="gru", anchor="ekf", d=32, epochs=32,
             lr=3e-3, use_cov=True, seed=0),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--augment-frac", type=float, default=0.5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(MODELDIR, exist_ok=True)
    print(f"generating {args.n_train}x{args.n_steps} imagination-curriculum training data "
          f"(augment_frac={args.augment_frac})")
    t0 = time.time()
    ds = data.generate_imagination_curriculum(args.n_train, args.n_steps, seed=11,
                                               augment_frac=args.augment_frac)
    val = data.generate(max(16, args.n_train // 12), args.n_steps, seed=12)  # standard val (unaugmented)
    print(f"data ready in {time.time()-t0:.1f}s")

    for spec in specs():
        path = os.path.join(MODELDIR, spec.name)
        if os.path.exists(path + ".npz") and not args.force:
            print(f"[skip] {spec.name}")
            continue
        print(f"[train] {spec.name}")
        t0 = time.time()
        params = torch_train.train(spec, ds, val, verbose=True)
        persist.save(path, spec, params)
        print(f"saved {spec.name} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
