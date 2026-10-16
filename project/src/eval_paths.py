
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .data import Bundle
from .planner import beam_search, candidate_set, greedy_path, make_hist, simulate_path
from .utils import get_device, load_config, save_json, set_seed
from .train import build


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--tag", default="full")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set); pc = cfg["planner"]
    set_seed(a.seed); dev = get_device()
    bundle, model = build(cfg, a.dataset, dev)
    run = Path("runs") / a.dataset / a.tag / f"seed{a.seed}"
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev, weights_only=True)); model.eval()
    L = pc["path_len"]
    sess = [s for s in bundle.sessions["test"] if len(s["items"]) >= L + 3][: pc["max_eval_learners"]]
    methods = {"proposed": [], "greedy_topL": [], "beam_no_smooth": []}
    times = []
    lam = pc["lambda_con"] if pc["use_smooth"] else 0.0
    for s in sess:
        cut = len(s["items"]) - L
        dt = np.concatenate([[0.0], np.diff(s["ts"])])[:cut]
        hist = make_hist(s["items"][:cut], s["correct"][:cut], dt, s["dur"][:cut], dev)
        tg_set = sorted({c for i in s["items"][cut:] for c in bundle.meta["clists"][i]})
        if not tg_set:
            continue
        targets = torch.tensor(tg_set, dtype=torch.long, device=dev)
        excl = set(s["items"][:cut].tolist())
        cands, plan = candidate_set(model, hist, targets, bundle.desc, bundle.concept_init, bundle.meta["clists"],
                                    bundle.meta["n_items"], pc["n_candidates"], pc["n_weak_concepts"], excl)
        t0 = time.time()
        path, _ = beam_search(model, hist, cands, targets, pc["beam_width"], L, lam, pc["eps"])
        times.append(time.time() - t0)
        methods["proposed"].append(simulate_path(model, hist, path, targets, pc["eps"]))
        methods["greedy_topL"].append(simulate_path(model, hist, greedy_path(model, hist, excl, L), targets, pc["eps"]))
        p0, _ = beam_search(model, hist, cands, targets, pc["beam_width"], L, 0.0, pc["eps"])
        methods["beam_no_smooth"].append(simulate_path(model, hist, p0, targets, pc["eps"]))
    res = {k: dict(Gp=float(np.mean([x[0] for x in v])), Rsmooth=float(np.mean([x[1] for x in v])), n=len(v)) for k, v in methods.items() if v}
    res["beam_search_sec_per_learner"] = float(np.mean(times)) if times else None
    res.update(seed=a.seed, tag=a.tag, dataset=a.dataset, lambda_con=lam)
    save_json(res, Path("results") / a.dataset / a.tag / f"seed{a.seed}_paths.json")
    print(res)


if __name__ == "__main__":
    main()
