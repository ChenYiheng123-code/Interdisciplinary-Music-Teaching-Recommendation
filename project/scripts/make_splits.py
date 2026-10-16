
import argparse

from musichg.data import build_sessions_and_splits
from musichg.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--config", default="configs/default.yaml")
a = ap.parse_args()
sessions, sp = build_sessions_and_splits(load_config(a.config), a.dataset)
print(f"{a.dataset}: {len(sessions)} sessions | train {len(sp['train'])} val {len(sp['val'])} test {len(sp['test'])} "
      f"dropped(straddling) {sp['n_dropped_straddling']} | sha256 {sp['sha256'][:16]}")
