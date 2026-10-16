
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import sha256_obj, save_json, load_json


def load_processed(root, ds):
    d = Path(root) / ds
    inter = pd.read_csv(d / "interactions.csv")
    items = pd.read_csv(d / "items.csv")
    conc = pd.read_csv(d / "concepts.csv")
    return inter, items, conc


def build_meta(items, conc):
    item_ids = items["item"].astype(str).tolist()
    c_ids = conc["concept"].astype(str).tolist()
    item2idx = {k: i for i, k in enumerate(item_ids)}
    c2idx = {k: i for i, k in enumerate(c_ids)}
    n, nc = len(item_ids), len(c_ids)
    clists = []
    for x in items["concepts"].tolist():
        cl = [c2idx[c] for c in str(x).split("|") if c in c2idx]
        clists.append(cl)
    cm = max(1, max(len(l) for l in clists))
    ic = np.full((n + 1, cm), nc, dtype=np.int64)  # row n = pad item, concept nc = pad concept
    for i, l in enumerate(clists):
        ic[i, : len(l)] = l
    return dict(item_ids=item_ids, c_ids=c_ids, item2idx=item2idx, c2idx=c2idx,
                n_items=n, n_concepts=nc, item_concepts=ic, clists=clists)


def sessionize(inter, item2idx, gap_days=30, max_nonresp=3, min_len=3):
    inter = inter.copy()
    inter["item_idx"] = inter["item"].astype(str).map(item2idx)
    inter = inter.dropna(subset=["item_idx"])
    if "responded" not in inter.columns:
        inter["responded"] = 1
    if "duration" not in inter.columns:
        inter["duration"] = np.nan
    inter = inter.sort_values(["user", "timestamp", "item"], kind="mergesort")
    gap = gap_days * 86400.0
    sessions = []
    for u, g in inter.groupby("user", sort=True):
        ts = g["timestamp"].values.astype(float)
        it = g["item_idx"].values.astype(int)
        cr = g["correct"].values.astype(float)
        du = g["duration"].values.astype(float)
        rs = g["responded"].values.astype(int)
        cur, nr, last_t, k = [], 0, None, 0

        def flush():
            nonlocal cur, k
            if len(cur) >= min_len:
                a = np.array(cur)
                t = ts[a]
                sessions.append(dict(sid=f"{u}_{k}", user=str(u), items=it[a], correct=cr[a],
                                     ts=t, dur=du[a]))
                k += 1
            cur = []

        for i in range(len(ts)):
            if rs[i] == 0:
                nr += 1
                if nr >= max_nonresp:
                    flush(); last_t = None; nr = 0
                continue
            nr = 0
            if last_t is not None and ts[i] - last_t > gap:
                flush()
            cur.append(i); last_t = ts[i]
        flush()
    return sessions


def make_splits(sessions, ratios=(0.8, 0.1, 0.1)):
    """Chronological split on interaction timestamps. Sessions that straddle a cut-off are dropped,
    so no train session contains information from after the train cut-off."""
    all_ts = np.sort(np.concatenate([s["ts"] for s in sessions]))
    t1 = float(all_ts[int(len(all_ts) * ratios[0])])
    t2 = float(all_ts[min(len(all_ts) - 1, int(len(all_ts) * (ratios[0] + ratios[1])))])
    sp = {"train": [], "val": [], "test": []}
    dropped = 0
    for s in sessions:
        st, en = s["ts"][0], s["ts"][-1]
        if en <= t1:
            sp["train"].append(s["sid"])
        elif st >= t1 and en <= t2:
            sp["val"].append(s["sid"])
        elif st >= t2:
            sp["test"].append(s["sid"])
        else:
            dropped += 1
    for k in sp:
        sp[k] = sorted(sp[k])
    sp["cutoffs"] = {"t_train_end": t1, "t_val_end": t2}
    sp["n_dropped_straddling"] = dropped
    sp["sha256"] = sha256_obj({k: sp[k] for k in ("train", "val", "test")})
    return sp


def item_stats(train_sessions, meta):
    n = meta["n_items"]
    err = np.zeros(n + 1); cnt = np.zeros(n + 1)
    ld = [[] for _ in range(n + 1)]
    for s in train_sessions:
        for i, c, d in zip(s["items"], s["correct"], s["dur"]):
            err[i] += 1 - c; cnt[i] += 1
            if not np.isnan(d):
                ld[i].append(np.log1p(max(d, 0)))
    err_rate = (err + 1.0) / (cnt + 2.0)  # Laplace smoothing
    nconc = np.array([len(l) for l in meta["clists"]] + [0], dtype=float)
    maxc = max(1.0, nconc.max())
    difficulty = err_rate * (nconc / maxc)
    allld = [x for l in ld for x in l]
    gm, gs = (float(np.mean(allld)), float(np.std(allld) + 1e-6)) if allld else (0.0, 1.0)
    mu = np.array([np.mean(l) if len(l) > 1 else gm for l in ld])
    sd = np.array([np.std(l) if len(l) > 1 else gs for l in ld])
    difficulty[n] = 0; mu[n] = gm; sd[n] = gs
    return dict(difficulty=difficulty.astype(np.float32), mu=mu.astype(np.float32), sd=sd.astype(np.float32))


def build_sessions_and_splits(cfg, ds):
    dc = cfg["data"]
    inter, items, conc = load_processed(dc["root"], ds)
    meta = build_meta(items, conc)
    sessions = sessionize(inter, meta["item2idx"], dc["session_gap_days"], dc["max_nonresponse"], dc["min_session_len"])
    sp = make_splits(sessions, tuple(dc["split_ratio"]))
    out = Path(dc["root"]) / ds
    with open(out / "sessions.pkl", "wb") as f:
        pickle.dump(sessions, f)
    save_json(sp, Path(dc["splits_dir"]) / f"{ds}.json")
    return sessions, sp


def _maybe(path):
    return np.load(path).astype(np.float32) if Path(path).exists() else None


class Bundle:
    """Everything a run needs for one dataset."""

    def __init__(self, cfg, ds):
        dc = cfg["data"]
        inter, items, conc = load_processed(dc["root"], ds)
        self.meta = build_meta(items, conc)
        with open(Path(dc["root"]) / ds / "sessions.pkl", "rb") as f:
            allsess = pickle.load(f)
        sp = load_json(Path(dc["splits_dir"]) / f"{ds}.json")
        by = {s["sid"]: s for s in allsess}
        self.split_hash = sp["sha256"]
        self.sessions = {k: [by[i] for i in sp[k]] for k in ("train", "val", "test")}
        self.stats = item_stats(self.sessions["train"], self.meta)
        fr = Path(dc["feature_root"]) / ds
        self.mod_feats = {m: _maybe(fr / f"{m}.npy") for m in ("vit", "audio", "text")}
        self.mod_feats = {k: v for k, v in self.mod_feats.items() if v is not None}
        self.concept_init = _maybe(fr / "concept_init.npy")
        self.desc = _maybe(fr / "desc.npy")
        self.items_df, self.conc_df = items, conc
        self.max_len = dc["max_len"]


class SeqDataset(Dataset):
    def __init__(self, sessions, max_len):
        self.s = sessions
        self.max_len = max_len

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        s = self.s[i]
        sl = slice(-self.max_len, None)
        ts = s["ts"][sl]
        dt = np.concatenate([[0.0], np.diff(ts)])
        dur = s["dur"][sl].copy()
        return dict(items=s["items"][sl], correct=s["correct"][sl], dt=dt, dur=dur, sid=s["sid"])


def make_collate(pad_item, dur_fill=30.0):
    def collate(batch):
        B = len(batch); T = max(len(b["items"]) for b in batch)
        items = torch.full((B, T), pad_item, dtype=torch.long)
        fb = torch.zeros(B, T); dt = torch.zeros(B, T); dur = torch.zeros(B, T)
        mask = torch.zeros(B, T, dtype=torch.bool)
        for i, b in enumerate(batch):
            n = len(b["items"])
            items[i, :n] = torch.from_numpy(b["items"])
            fb[i, :n] = torch.from_numpy(b["correct"]).float()
            dt[i, :n] = torch.from_numpy(b["dt"]).float()
            d = np.nan_to_num(b["dur"], nan=dur_fill)
            dur[i, :n] = torch.from_numpy(d).float()
            mask[i, :n] = True
        return dict(items=items, fb=fb, dt=dt, dur=dur, mask=mask)
    return collate
