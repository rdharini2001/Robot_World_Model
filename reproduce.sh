#!/usr/bin/env bash
# Reproduces every new table and figure in the workshop paper
# (paper/main.pdf), on top of the original ECE 6562 pipeline.
#
# Usage:
#   ./reproduce_worldmodel.sh            # use bundled checkpoints + results
#   ./reproduce_worldmodel.sh --retrain  # retrain the imagination arm from scratch
#   ./reproduce_worldmodel.sh --full     # retrain + rerun the full evaluation + rebuild the PDF
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
export OMP_NUM_THREADS=1

RETRAIN=0
FULL=0
for arg in "$@"; do
  case "$arg" in
    --retrain) RETRAIN=1 ;;
    --full) FULL=1; RETRAIN=1 ;;
    *) echo "Unknown option: $arg"; exit 2 ;;
  esac
done

echo "[1/5] Running the original test suite (unchanged core pipeline)"
pytest -q

if [[ "$RETRAIN" == "1" ]]; then
  echo "[2/5] Training the imagination-curriculum observer arm (~6 min on CPU)"
  python scripts/train_rollout_curriculum.py --force
else
  echo "[2/5] Using bundled imagination-curriculum checkpoints (results/models_worldmodel/)"
fi

if [[ "$FULL" == "1" ]]; then
  echo "[3/5] Rerunning the full world-model evaluation (~7 min on CPU)"
  rm -rf results/worldmodel
  python scripts/evaluate_worldmodel.py
  echo "[4/5] Regenerating figures"
  python scripts/make_figures.py
  cp figures/worldmodel/*.png paper/figures/
  echo "[4b/5] Recompiling the paper"
  (cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && \
               bibtex main >/dev/null && \
               pdflatex -interaction=nonstopmode main.tex >/dev/null && \
               pdflatex -interaction=nonstopmode main.tex >/dev/null)
  mv paper/main.pdf paper/paper.pdf
else
  echo "[3/5] Using bundled results/worldmodel/*.json"
  echo "[4/5] Using bundled figures/worldmodel/*.png and paper/paper.pdf"
fi

echo "[5/5] Summary of key numbers"
python - <<'PY'
import json
d = json.load(open("results/worldmodel/proxy_with_imagination.json"))
print("Offline selection criteria, 24 conditions, 6 core observers:")
for k in ["position_rmse", "heading_rmse", "imagine_rmse_H20"]:
    v = d[k]
    print(f"  {k:20s} spearman={v['spearman']:.3f} flip={v['flip_fraction']:.3f} "
          f"mean_regret={v['mean_regret']:.4f} max_regret={v['max_regret']:.4f}")

c = json.load(open("results/worldmodel/wm_compound.json"))
print("\nCompound stress, cross-track RMSE (m):")
for name in ["ekf", "ssm_ekf", "ssm_ekf_imagine_seed0", "gru_ekf", "gru_ekf_imagine_seed0",
             "ssm_dr", "ssm_dr_imagine_seed0"]:
    print(f"  {name:26s} {c[name]['ct_rmse'][0]:.3f}")
PY

echo ""
echo "Done. Open paper/paper.pdf."
