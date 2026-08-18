#!/usr/bin/env python3
"""Check the files and result values required for the course submission."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "requirements.txt",
    "environment.yml",
    "report/ECE6562_Final_Project_Report.pdf",
    "report/ECE6562_Final_Project_Report.tex",
    "demo/VIDEO_SCRIPT.md",
    "demo/nominal_tracking.mp4",
    "demo/blackout_tracking.mp4",
    "results/reference/nominal.json",
    "results/reference/enhanced/proxy_analysis.json",
    "results/models/gru_ekf.npz",
    "results/models/ssm_ekf.npz",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        print("Missing required files:")
        for name in missing:
            print(" -", name)
        raise SystemExit(1)

    nominal = json.loads((ROOT / "results/reference/nominal.json").read_text())
    proxy = json.loads((ROOT / "results/reference/enhanced/proxy_analysis.json").read_text())
    expected_observers = {"dead_reckoning", "ekf", "gru_dr", "ssm_dr", "gru_ekf", "ssm_ekf"}
    if not expected_observers.issubset(nominal):
        raise RuntimeError("Nominal result table is incomplete")
    pose = proxy["analysis"]["pose_rmse"]
    if len(proxy["conditions"]) != 24 or len(proxy["observers"]) != 6:
        raise RuntimeError("Selection analysis must contain 24 conditions and 6 observers")
    if abs(pose["selection"]["flip_fraction"] - 5 / 24) > 1e-12:
        raise RuntimeError("Unexpected top-1 selection result")

    manifest = ROOT / "SUBMISSION_MANIFEST.sha256"
    files = [ROOT / name for name in REQUIRED]
    manifest.write_text("\n".join(f"{sha256(path)}  {path.relative_to(ROOT)}" for path in files) + "\n")

    print("Submission check passed")
    print(f"  required files: {len(REQUIRED)}")
    print(f"  observers: {len(proxy['observers'])}")
    print(f"  operating conditions: {len(proxy['conditions'])}")
    print(f"  replay-RMSE top-1 failures: 5/24 ({100*5/24:.1f}%)")
    print(f"  report size: {(ROOT/'report/ECE6562_Final_Project_Report.pdf').stat().st_size/1e6:.2f} MB")
    print(f"  manifest: {manifest.name}")


if __name__ == "__main__":
    main()
