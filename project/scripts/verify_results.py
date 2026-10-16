import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, help="our dataset dir name under results/")
ap.add_argument("--paper-key", default=None, help="key in paper_reported.json (default = --dataset)")
ap.add_argument("--tag", default="full"); ap.add_argument("--tol", type=float, default=0.01)
a = ap.parse_args()
P = json.load(open("paper_reported.json")); key = a.paper_key or a.dataset
S = json.load(open(Path("results") / a.dataset / a.tag / "summary.json"))
rows = []
if key in P["tracing"]:
    for i, m in enumerate(["AUC", "ACC", "RMSE"]):
        rows.append((m, P["tracing"][key]["Proposed"][i], S.get(m)))
if key in P["paths"]:
    for i, m in enumerate(["Recall@10", "NDCG@10"]):
        rows.append((m, P["paths"][key]["Proposed"][i], S.get(m)))
    rows.append(("Gp", P["paths"][key]["Proposed"][2], S.get("Gp[proposed]")))
    rows.append(("Rsmooth", P["paths"][key]["Proposed"][3], S.get("Rsmooth[proposed]")))
print(f"{'metric':12s} {'paper':>8s} {'ours(mean±std)':>18s} {'|Δ|':>8s}  within tol={a.tol}")
for m, pv, o in rows:
    if o is None:
        print(f"{m:12s} {pv:8.4f} {'n/a':>18s}"); continue
    d = abs(pv - o["mean"])
    print(f"{m:12s} {pv:8.4f} {o['mean']:9.4f}±{o['std']:.4f} {d:8.4f}  {'YES' if d <= a.tol else 'no'}")
print("\nNOTE: Rsmooth/Gp are only comparable if the embedding scale and gain definition match the paper's (unspecified); see docs/DEVIATIONS.md.")
