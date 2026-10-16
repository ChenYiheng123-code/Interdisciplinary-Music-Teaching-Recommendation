
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .llm_agent import DummyLLM, QwenLLM, explain_once
from .memory import DIM_NAMES, ExperienceStore, Memory
from .planner import beam_search, candidate_set, make_hist, simulate_path
from .train import build
from .utils import get_device, load_config, load_json, set_seed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--tag", default="full")
    ap.add_argument("--llm", default="dummy", choices=["dummy", "qwen"]); ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set); pc = cfg["planner"]
    set_seed(a.seed); dev = get_device()
    bundle, model = build(cfg, a.dataset, dev)
    run = Path("runs") / a.dataset / a.tag / f"seed{a.seed}"
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev, weights_only=True)); model.eval()
    if a.llm == "qwen":
        lock = load_json("models.lock.json") if Path("models.lock.json").exists() else {}
        llm = QwenLLM(revision=lock.get("llm", {}).get("revision"), device=str(dev))
    else:
        llm = DummyLLM()
    store = ExperienceStore(pc["beta_value"], pc["value_lr"]); L = pc["path_len"]
    names = bundle.items_df["title"].astype(str).tolist(); cnames = bundle.conc_df["name"].astype(str).tolist()
    out_f = Path("results") / a.dataset / a.tag / f"seed{a.seed}_agent.jsonl"
    out_f.parent.mkdir(parents=True, exist_ok=True)
    sess = [s for s in bundle.sessions["test"] if len(s["items"]) >= L + 3][: a.n]
    with open(out_f, "w") as fo:
        for s in sess:
            cut = len(s["items"]) - L
            dt = np.concatenate([[0.0], np.diff(s["ts"])])[:cut]
            hist = make_hist(s["items"][:cut], s["correct"][:cut], dt, s["dur"][:cut], dev)
            tg = sorted({c for i in s["items"][cut:] for c in bundle.meta["clists"][i]})
            if not tg:
                continue
            targets = torch.tensor(tg, dtype=torch.long, device=dev)
            excl = set(s["items"][:cut].tolist())
            t0 = time.time()
            cands, plan = candidate_set(model, hist, targets, bundle.desc, bundle.concept_init, bundle.meta["clists"],
                                        bundle.meta["n_items"], pc["n_candidates"], pc["n_weak_concepts"], excl)
            path, info = beam_search(model, hist, cands, targets, pc["beam_width"], L, pc["lambda_con"] if pc["use_smooth"] else 0.0, pc["eps"])
            t_search = time.time() - t0
            mem = Memory(bias_threshold=pc["bias_threshold"], min_events=pc["bias_min_events"])
            corr = [c for c in (mem.correction(c, cnames[c]) for c in tg) if c]
            t1 = time.time()
            rep, fb = explain_once(llm, [names[i] for i in path], {"gain": info["G"], "smooth": info["R"]},
                                   "target: " + ", ".join(cnames[c] for c in tg), corr, DIM_NAMES)
            t_llm = time.time() - t1
            G, R = simulate_path(model, hist, path, targets, pc["eps"])
            store.add(np.zeros(8), f"path gain {G:.3f}", G)
            fo.write(json.dumps(dict(sid=s["sid"], path=path, titles=[names[i] for i in path], Gp=G, Rsmooth=R, report=rep, used_fallback=fb,
                                     t_search_s=t_search, t_llm_s=t_llm), ensure_ascii=False) + "\n")
    print("wrote", out_f)


if __name__ == "__main__":
    main()
