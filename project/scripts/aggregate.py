
import argparse
import glob
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--tag", default="full")
a = ap.parse_args()
d = Path("results") / a.dataset / a.tag
tr = [json.load(open(f)) for f in sorted(glob.glob(str(d / "seed*_tracing.json")))]
pa = [json.load(open(f)) for f in sorted(glob.glob(str(d / "seed*_paths.json")))]
agg = lambda xs: dict(mean=float(np.mean(xs)), std=float(np.std(xs)), n=len(xs))
S = {"seeds": [t["seed"] for t in tr]}
for k in ("AUC", "ACC", "RMSE", "Recall@10", "NDCG@10"):
    if tr: S[k] = agg([t["test"][k] for t in tr])
for m in ("proposed", "greedy_topL", "beam_no_smooth"):
    for k, name in (("Gp", f"Gp[{m}]"), ("Rsmooth", f"Rsmooth[{m}]")):
        xs = [p[m][k] for p in pa if m in p]
        if xs: S[name] = agg(xs)
json.dump(S, open(d / "summary.json", "w"), indent=2)
print(json.dumps(S, indent=2))
