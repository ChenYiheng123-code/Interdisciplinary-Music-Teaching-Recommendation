
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hypergraph import HypergraphEncoder
from .moe import MoEHub
from .ssm import MambaBlock


def _init_concepts(concept_init, nc, d, seed_gen):
    w = torch.randn(nc + 1, d, generator=seed_gen) * 0.1
    if concept_init is not None:
        X = torch.from_numpy(concept_init)
        X = X - X.mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        r = min(d, Vh.shape[0])
        Y = X @ Vh[:r].T
        Y = Y / (Y.std() + 1e-6) * 0.1
        w[:nc, :r] = Y
    w[nc] = 0
    return w


class CognitiveTracer(nn.Module):
    def __init__(self, cfg, bundle):
        super().__init__()
        mc = cfg["model"]
        self.mc = mc
        d = mc["d"]; self.d = d
        meta = bundle.meta
        self.n, self.nc = meta["n_items"], meta["n_concepts"]
        self.hg = HypergraphEncoder(self.n, d, bundle.mod_feats, meta["clists"], self.nc, mc["hg_k"], mc["hg_layers"],
                                    mc["use_concept_hyperedges"], mc["use_degree_norm"], mc["use_modal_residual"], mc["dropout"])
        g = torch.Generator().manual_seed(1234)
        self.concept_emb = nn.Embedding(self.nc + 1, d, padding_idx=self.nc)   # independent dictionary (no graph gradient)
        self.concept_emb.weight.data.copy_(_init_concepts(bundle.concept_init, self.nc, d, g))
        self.register_buffer("ic", torch.from_numpy(meta["item_concepts"]))
        for k in ("difficulty", "mu", "sd"):
            self.register_buffer(k, torch.from_numpy(bundle.stats[k]))
        self.fb_emb = nn.Embedding(2, d)
        self.time_emb = nn.Embedding(mc["time_buckets"], d)
        self.inp = nn.Linear(4 * d, d)
        self.max_pos = bundle.max_len + 64
        if mc["encoder"] == "mamba":
            self.fwd = MambaBlock(d, mc["d_state"], mc["d_conv"])
            self.bwd = MambaBlock(d, mc["d_state"], mc["d_conv"]) if mc["use_backward"] else None
            self.mix = nn.Linear(2 * d, d)
        else:
            self.pos = nn.Embedding(self.max_pos, d)
            layer = nn.TransformerEncoderLayer(d, mc["n_heads"], 2 * d, mc["dropout"], batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
        self.norm = nn.LayerNorm(d)
        self.s0 = nn.Parameter(torch.randn(1, 1, d) * 0.1)             # learnable initial state prototype
        self.moe = MoEHub(d, mc["ctx_window_wc"], mc["moe_topk"], mc["n_heads"], mc["use_causal_attn_expert"], mc["dropout"]) if mc["use_moe"] else None
        self.head = nn.Sequential(nn.Linear(3 * d if self.moe else 2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.mast = nn.Linear(d, d)                                    # state -> concept space (Eq. 36)
        self.q = nn.Linear(d, d)                                       # preference query for BPR / ranking

    # ---------- helpers
    def item_table(self):
        H = self.hg()
        return torch.cat([H, H.new_zeros(1, self.d)], 0), H

    def bucket(self, dt):
        return torch.clamp(torch.log2(1 + dt / 60.0).floor().long(), 0, self.mc["time_buckets"] - 1)

    def mastery_all(self, S):
        return torch.sigmoid(self.mast(S) @ self.concept_emb.weight[: self.nc].T / math.sqrt(self.d))

    def mastery_of(self, S, cids):
        """S [B,d]; cids LongTensor [K] -> [B,K]"""
        return torch.sigmoid(self.mast(S) @ self.concept_emb(cids).T / math.sqrt(self.d))

    # ---------- diagnostics (history-only; no look-ahead)
    def diag_feats(self, items, fb, dur, mask, S_prev, e):
        mc = self.mc; B, T = items.shape
        cid = self.ic[items]
        f1 = self.difficulty[items]
        with torch.no_grad():
            m_all = self.mastery_all(S_prev)
            weak = m_all.argmin(-1)
            f2 = F.cosine_similarity(e.detach(), self.concept_emb(weak), -1)
            z = ((torch.log1p(dur) - self.mu[items]) / (self.sd[items] + 1e-6)).clamp(-5, 5) * mask
            f3 = F.pad(z, (1, 0))[:, :T]                               # duration of previous step only
            num = torch.zeros_like(f1); den = torch.zeros_like(f1)
            for j in range(1, mc["err_lookback"] + 1):
                it_sh = F.pad(items, (j, 0), value=self.n)[:, :T]
                fb_sh = F.pad(fb, (j, 0), value=1.0)[:, :T]
                cs = self.ic[it_sh]
                share = ((cid.unsqueeze(-1) == cs.unsqueeze(-2)) & (cid.unsqueeze(-1) != self.nc)).flatten(2).any(-1).float()
                w = (mc["err_decay"] ** j) * share
                num += w * (1 - fb_sh); den += w
            f4 = num / (den + 1e-6)
            prim = cid[..., 0].clamp(max=self.nc - 1).unsqueeze(-1)
            mp = m_all.gather(-1, prim).squeeze(-1)
            mp_prev = torch.cat([mp[:, :1], mp[:, :-1]], 1)
            f5 = mp - math.exp(-mc["gain_forget"]) * mp_prev
        return torch.stack([f1, f2, f3, f4, f5], -1)

    # ---------- main forward
    def forward(self, b, table=None):
        items, fb, dt, dur, mask = b["items"], b["fb"].float(), b["dt"], b["dur"], b["mask"]
        E, H = table if table is not None else self.item_table()
        B, T = items.shape
        e = E[items]
        cid = self.ic[items]
        cm = (cid != self.nc).float().unsqueeze(-1)
        ce = (self.concept_emb(cid) * cm).sum(2) / cm.sum(2).clamp(min=1)              # Eq. 16
        fbe = fb.unsqueeze(-1) * self.fb_emb.weight[1] + (1 - fb.unsqueeze(-1)) * self.fb_emb.weight[0]
        te = self.time_emb(self.bucket(dt)) if self.mc["use_time_emb"] else torch.zeros_like(e)
        x = self.inp(torch.cat([e, fbe, te, ce], -1))                                  # Eq. 17
        if self.mc["encoder"] == "mamba":
            hf = self.fwd(x, mask)
            hb = self.bwd.forward_window(x, self.mc["window_w"]) if self.bwd is not None else torch.zeros_like(hf)
            S = self.norm(self.mix(torch.cat([hf, hb], -1)))                            # Eq. 22
        else:
            causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
            pos = torch.arange(T, device=x.device).clamp(max=self.max_pos - 1)
            S = self.norm(self.enc(x + self.pos(pos), mask=causal))
        S_prev = torch.cat([self.s0.expand(B, 1, -1), S[:, :-1]], 1)                    # state BEFORE each step
        out = dict(S=S, S_prev=S_prev, e=e, H=H, E=E)
        if self.moe is not None:
            feats = self.diag_feats(items, fb, dur, mask.float(), S_prev, e)
            diag, p_dense, p_sparse = self.moe(feats)
            out.update(diag=diag, gate_dense=p_dense, gate_sparse=p_sparse, feats=feats)
            z = torch.cat([S_prev, e, diag], -1)
        else:
            z = torch.cat([S_prev, e], -1)
        out["logit"] = self.head(z).squeeze(-1)
        out["pref_q"] = self.q(S_prev)
        return out
