
import argparse
import glob
import hashlib
from pathlib import Path

from musichg.utils import env_info, save_json, sha256_file

ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); a = ap.parse_args()
files = {}
for pat in [f"data/processed/{a.dataset}/*.csv", f"data/features/{a.dataset}/*.npy", f"splits/{a.dataset}.json", "configs/*",
            "prompts/*", "models.yaml", "models.lock.json", "src/musichg/*.py", "requirements.txt"]:
    for f in sorted(glob.glob(pat)):
        files[f] = sha256_file(f)
save_json(dict(dataset=a.dataset, env=env_info(), sha256=files), Path("results") / a.dataset / "MANIFEST.json")
print(f"manifest: {len(files)} files hashed")
