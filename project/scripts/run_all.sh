#!/usr/bin/env bash
# Full protocol for ONE dataset over all seeds in configs/seeds.json.
#   bash scripts/run_all.sh <dataset> [config]
# Prerequisites: data/processed/<dataset>/ prepared, features extracted (if multimodal), models.lock.json pinned.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
DS=${1:?dataset}; CFG=${2:-configs/default.yaml}
SEEDS=$(python -c "import json;print(' '.join(map(str,json.load(open('configs/seeds.json'))['seeds'])))")
python scripts/make_splits.py --dataset "$DS" --config "$CFG"
for S in $SEEDS; do
  python -m musichg.train      --dataset "$DS" --seed "$S" --tag full --config "$CFG"
  python -m musichg.eval_paths --dataset "$DS" --seed "$S" --tag full --config "$CFG"
done
python scripts/aggregate.py --dataset "$DS" --tag full
python scripts/make_manifest.py --dataset "$DS"
