
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Bundle, SeqDataset, make_collate
from .losses import bce_loss, bpr_loss, infonce_loss, max_entropy_loss
from .metrics import ranking_hits, tracing_metrics
from .tracer import CognitiveTracer
from .utils import env_info, get_device, load_config, save_json, set_seed


def to_dev(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, loader, dev, n_items, k=10):
    model.eval()
    ys, ps, hits, ndcg = [], [], [], []
    for b in loader:
        b = to_dev(b, dev)
        out = model(b)
        m = b["mask"]
        ys.append(b["fb"][m].cpu()); ps.append(torch.sigmoid(out["logit"])[m].cpu())
        mm = m.clone(); mm[:, 0] = False
        h, n = ranking_hits(out["pref_q"][mm], out["E"][:n_items], b["items"][mm], k)
        hits.append(h); ndcg.append(n)
    r = tracing_metrics(torch.cat(ys).numpy(), torch.cat(ps).numpy())
    r["Recall@10"] = float(np.concatenate(hits).mean()); r["NDCG@10"] = float(np.concatenate(ndcg).mean())
    return r


def build(cfg, ds, dev):
    bundle = Bundle(cfg, ds)
    model = CognitiveTracer(cfg, bundle).to(dev)
    return bundle, model


def loaders(cfg, bundle, seed):
    pad = bundle.meta["n_items"]
    col = make_collate(pad)
    g = torch.Generator().manual_seed(seed)
    mk = lambda s, sh: DataLoader(SeqDataset(bundle.sessions[s], bundle.max_len), batch_size=cfg["train"]["batch_size"],
                                  shuffle=sh, collate_fn=col, generator=g, num_workers=cfg["train"]["num_workers"])
    return mk("train", True), mk("val", False), mk("test", False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    tc = cfg["train"]
    set_seed(a.seed)
    dev = get_device()
    bundle, model = build(cfg, a.dataset, dev)
    tr, va, te = loaders(cfg, bundle, a.seed)
    run = Path("runs") / a.dataset / a.tag / f"seed{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    save_json(dict(cfg=cfg, seed=a.seed, split_sha256=bundle.split_hash, env=env_info(),
                   n_params=sum(p.numel() for p in model.parameters())), run / "meta.json")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total = tc["max_epochs"]; wu = tc["warmup_epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / wu if e < wu else 0.5 * (1 + math.cos(math.pi * (e - wu) / max(1, total - wu))))
    n_items = bundle.meta["n_items"]
    best, bad, log = -1e9, 0, []
    for ep in range(total):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for b in tr:
            b = to_dev(b, dev)
            out = model(b)
            m = b["mask"].float()
            loss = bce_loss(out["logit"], b["fb"], m)
            loss = loss + tc["lambda_rank"] * bpr_loss(out, b["items"], b["mask"], n_items)
            if tc["use_cl"]:
                loss = loss + tc["lambda_cl"] * infonce_loss(out["H"], model.hg.id_emb.weight, b["items"], tc["tau"], tc["cl_max_items"], n_items)
            if tc["use_entropy_reg"] and "gate_dense" in out:
                loss = loss + tc["lambda_ent"] * max_entropy_loss(out["gate_dense"], b["mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step(); tot += loss.item(); nb += 1
        sched.step()
        v = evaluate(model, va, dev, n_items)
        crit = v["AUC"] + v["NDCG@10"]            # monitors cognitive prediction + preference ranking
        log.append(dict(epoch=ep, loss=tot / max(1, nb), val=v, sec=time.time() - t0))
        print(f"ep{ep:03d} loss {tot / max(1, nb):.4f} val AUC {v['AUC']:.4f} NDCG@10 {v['NDCG@10']:.4f}", flush=True)
        if crit > best:
            best, bad = crit, 0
            torch.save(model.state_dict(), run / "best.pt")
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    save_json(log, run / "train_log.json")
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev, weights_only=True))
    res = dict(val=evaluate(model, va, dev, n_items), test=evaluate(model, te, dev, n_items), seed=a.seed, tag=a.tag, dataset=a.dataset)
    save_json(res, Path("results") / a.dataset / a.tag / f"seed{a.seed}_tracing.json")
    print("TEST", res["test"])


if __name__ == "__main__":
    main()
