
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/synthetic"); ap.add_argument("--feat", default="data/features/synthetic")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--users", type=int, default=150)
    ap.add_argument("--concepts_per_domain", type=int, default=6); ap.add_argument("--items_per_concept", type=int, default=3)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    D, cpd, ipc = 3, a.concepts_per_domain, a.items_per_concept
    nc = D * cpd
    dom = np.repeat(np.arange(D), cpd)
    cn = ["music_theory", "acoustics", "math"]
    concepts = pd.DataFrame(dict(concept=[f"c{i}" for i in range(nc)], name=[f"{cn[dom[i]]}_{i % cpd}" for i in range(nc)]))
    rows = []
    for c in range(nc):
        for k in range(ipc):
            cl = [c] + ([c - 1] if c % cpd > 0 and rng.random() < 0.4 else []) + ([(c + cpd) % nc] if rng.random() < 0.15 else [])
            rows.append(dict(item=f"i{len(rows)}", title=f"{cn[dom[c]]} exercise {c}-{k}", attrs=f"level{c % cpd}",
                             concepts="|".join(f"c{x}" for x in sorted(set(cl)))))
    items = pd.DataFrame(rows); n = len(items)
    prim = np.array([int(r.split("|")[0][1:]) for r in items["concepts"]])
    diff = (prim % cpd) / cpd
    cent = {m: rng.normal(size=(D, dim)) for m, dim in (("vit", 32), ("audio", 24), ("text", 48))}
    Path(a.feat).mkdir(parents=True, exist_ok=True)
    for m, c in cent.items():
        np.save(Path(a.feat) / f"{m}.npy", (c[dom[prim]] + 0.5 * rng.normal(size=(n, c.shape[1]))).astype(np.float32))
    cinit = rng.normal(size=(nc, 16)); cinit += 2 * np.eye(D)[dom] @ rng.normal(size=(D, 16))
    np.save(Path(a.feat) / "concept_init.npy", cinit.astype(np.float32))
    inter = []
    t0 = 1.6e9
    for u in range(a.users):
        know = rng.normal(-1.0, 0.5, size=nc); t = t0 + rng.uniform(0, 3e7)
        for _ in range(int(rng.integers(20, 60))):
            lvl = np.clip(np.round(know.mean() * 2 + rng.integers(0, 3)), 0, cpd - 1)
            cand = np.where((prim % cpd) <= lvl + 1)[0]
            i = int(rng.choice(cand)); c = prim[i]
            p = 1 / (1 + np.exp(-(know[c] - 2.5 * diff[i])))
            y = int(rng.random() < p)
            if y: know[c] += 0.4
            know *= 0.995
            t += float(rng.exponential(3600 * 6))
            inter.append(dict(user=f"u{u}", item=f"i{i}", correct=y, timestamp=t, duration=float(rng.lognormal(3, 0.5 + 0.5 * (1 - y)))))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inter).to_csv(Path(a.out) / "interactions.csv", index=False)
    items.to_csv(Path(a.out) / "items.csv", index=False); concepts.to_csv(Path(a.out) / "concepts.csv", index=False)
    print(f"synthetic: {a.users} users, {n} items, {nc} concepts, {len(inter)} interactions")


if __name__ == "__main__":
    main()
