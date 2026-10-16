
import torch
import torch.nn as nn
import torch.nn.functional as F


class DimExpert(nn.Module):
    def __init__(self, d, n_heads, wc, use_attn, dropout):
        super().__init__()
        self.wc, self.use_attn = wc, use_attn
        self.proj = nn.Linear(1, d)                                  # Eq. 23
        self.ffn = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True) if use_attn else None
        self.branch = nn.Linear(2, 2)
        self.norm = nn.LayerNorm(d)

    def forward(self, f, band_mask):
        # f: [B,T]
        h = self.proj(f.unsqueeze(-1))
        pad = F.pad(f.unsqueeze(1), (self.wc - 1, 0))
        wmean = F.avg_pool1d(pad, self.wc, 1).squeeze(1)              # causal window mean
        g = torch.softmax(self.branch(torch.stack([f, wmean], -1)), -1)   # [B,T,2]
        o = g[..., 0:1] * self.ffn(h)
        if self.use_attn:
            a, _ = self.attn(h, h, h, attn_mask=band_mask, need_weights=False)
            o = o + g[..., 1:2] * a
        return self.norm(h + o)                                       # Eq. 25


class MoEHub(nn.Module):
    def __init__(self, d, wc=5, topk=3, n_heads=4, use_attn=True, dropout=0.1, n_dims=5):
        super().__init__()
        self.n_dims, self.topk, self.wc = n_dims, topk, wc
        self.experts = nn.ModuleList([DimExpert(d, n_heads, wc, use_attn, dropout) for _ in range(n_dims)])
        self.gate = nn.Linear(n_dims * d, n_dims)

    def band_mask(self, T, device):
        i = torch.arange(T, device=device)
        diff = i[:, None] - i[None, :]
        return (diff < 0) | (diff >= self.wc)                         # True = masked (future or beyond window)

    def forward(self, feats):
        # feats [B,T,5]
        T = feats.shape[1]
        bm = self.band_mask(T, feats.device)
        reps = torch.stack([e(feats[..., k], bm) for k, e in enumerate(self.experts)], 2)   # [B,T,5,d]
        p = torch.softmax(self.gate(reps.flatten(2)), -1)             # dense importance distribution
        k = min(self.topk, self.n_dims)
        top = p.topk(k, -1)
        sparse = torch.zeros_like(p).scatter(-1, top.indices, top.values)
        sparse = sparse / sparse.sum(-1, keepdim=True).clamp(min=1e-9)
        fused = (sparse.unsqueeze(-1) * reps).sum(2)                  # Eq. 27
        return fused, p, sparse
