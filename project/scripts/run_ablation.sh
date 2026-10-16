#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
DS=${1:?dataset}; CFG=${2:-configs/default.yaml}
SEEDS=$(python -c "import json;print(' '.join(map(str,json.load(open('configs/seeds.json'))['seeds'])))")
python - "$DS" "$CFG" "$SEEDS" <<'PY'
import subprocess, sys, yaml
ds, cfg, seeds = sys.argv[1], sys.argv[2], sys.argv[3].split()
abl = yaml.safe_load(open("configs/ablations.yaml"))
for tag, sets in abl.items():
    for s in seeds:
        base = ["--dataset", ds, "--seed", s, "--tag", tag, "--config", cfg, "--set", *sets]
        subprocess.run([sys.executable, "-m", "musichg.train", *base], check=True)
        subprocess.run([sys.executable, "-m", "musichg.eval_paths", *base], check=True)
    subprocess.run([sys.executable, "scripts/aggregate.py", "--dataset", ds, "--tag", tag], check=True)
PY
