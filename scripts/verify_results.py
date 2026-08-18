#!/usr/bin/env python3
"""Verify the headline numerical claims reported in the workshop paper."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "worldmodel"

def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) <= tol

def main():
    proxy = json.loads((RES / "proxy_with_imagination.json").read_text())
    assert proxy["n_conditions"] == 24
    assert close(proxy["position_rmse"]["spearman"], 0.923, 5e-4)
    assert close(proxy["imagine_rmse_H20"]["spearman"], 0.774, 5e-4)
    assert close(proxy["position_rmse"]["flip_fraction"], 5/24, 1e-12)
    assert close(proxy["imagine_rmse_H20"]["flip_fraction"], 18/24, 1e-12)
    assert close(proxy["position_rmse"]["max_regret"], 0.0347, 5e-5)
    assert close(proxy["imagine_rmse_H20"]["max_regret"], 0.1208, 5e-5)

    compound = json.loads((RES / "wm_compound.json").read_text())
    expected = {
        "ssm_ekf": 1.936,
        "ssm_ekf_imagine_seed0": 1.419,
        "gru_ekf": 1.717,
        "gru_ekf_imagine_seed0": 1.061,
        "ssm_dr": 5.290,
        "ssm_dr_imagine_seed0": 5.411,
    }
    for name, value in expected.items():
        assert close(compound[name]["ct_rmse"][0], value, 5e-4), (name, compound[name]["ct_rmse"][0])

    sweep = json.loads((RES / "wm_sweep_dropout_len.json").read_text())
    assert sweep["_levels"] == [4, 10, 16, 22, 28, 34]
    assert close(sweep["gru_ekf"][4]["ct_rmse"][0], 0.286, 5e-4)
    assert close(sweep["gru_ekf_imagine_seed0"][4]["ct_rmse"][0], 0.183, 5e-4)
    assert close(sweep["ssm_ekf"][0]["ct_rmse"][0], 0.166, 5e-4)
    assert close(sweep["ssm_ekf_imagine_seed0"][0]["ct_rmse"][0], 0.233, 5e-4)
    assert close(sweep["ssm_ekf"][5]["ct_rmse"][0], 0.336, 5e-4)
    assert close(sweep["ssm_ekf_imagine_seed0"][5]["ct_rmse"][0], 0.819, 5e-4)

    print("All headline paper claims match the bundled result JSON files.")
    print("  24 conditions / 6 core observers / 144 observer-condition pairs")
    print("  Replay rho: 0.923; blind-rollout rho: 0.774")
    print("  Top-1 failures: 5/24 vs 18/24")
    print("  GRU-EKF compound stress: 1.717 -> 1.061 m")
    print("  SSM-EKF compound stress: 1.936 -> 1.419 m")

if __name__ == "__main__":
    main()
