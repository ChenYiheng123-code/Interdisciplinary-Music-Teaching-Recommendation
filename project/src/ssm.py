
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan(u, delta, A, B, C, D, mask=None):
    """u,delta:[N,L,Di]  A:[Di,S]  B,C:[N,L,S]  mask:[N,L] (1 = valid step, 0 = state/out unchanged). Returns [N,L,Di]."""
    N, L, Di = u.shape
    h = u.new_zeros(N, Di, A.shape[1])
    y_prev = u.new_zeros(N, Di)
    ys = []
    for t in range(L):
        dA = torch.exp(delta[:, t, :, None] * A[None])
        dBu = delta[:, t, :, None] * B[:, t, None, :] * u[:, t, :, None]
        if mask is not None:
            m = mask[:, t, None, None].to(u.dtype)
            dA = dA * m + (1 - m)
            dBu = dBu * m
        h = dA * h + dBu
        y = (h * C[:, t, None, :]).sum(-1)
        if mask is not None:
            mm = mask[:, t, None].to(u.dtype)
            y = mm * y + (1 - mm) * y_prev
        y_prev = y
        ys.append(y)
    return torch.stack(ys, 1) + u * D


class MambaBlock(nn.Module):
    def __init__(self, d, d_state=8, d_conv=4, expand=1):
        super().__init__()
        di = d * expand
        self.di, self.S = di, d_state
        self.dt_rank = max(1, math.ceil(d / 16))
        self.in_proj = nn.Linear(d, 2 * di)
        self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        self.x_proj = nn.Linear(di, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d)

    def _pre(self, x):
        L = x.shape[1]
        u, z = self.in_proj(x).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))   # causal conv
        dt, B, C = self.x_proj(u).split([self.dt_rank, self.S, self.S], -1)
        return u, z, F.softplus(self.dt_proj(dt)), B, C

    def forward(self, x, mask=None):
        u, z, delta, B, C = self._pre(x)
        y = selective_scan(u, delta, -torch.exp(self.A_log), B, C, self.D, mask)
        return self.out_proj(y * F.silu(z))

    def forward_window(self, x, w):
        """Restricted backward pathway: for every t scan x_t, x_{t-1}, ..., x_{t-w+1} (reverse order, look-back only,
        never touches t+1..). Adaptive truncation for t < w-1 via masking. Returns terminal output per t."""
        Bsz, T, _ = x.shape
        u, z, delta, Bm, Cm = self._pre(x)
        t = torch.arange(T, device=x.device)[:, None]
        j = torch.arange(w, device=x.device)[None, :]
        src = t - j                                   # [T,w] reverse-time indices
        valid = (src >= 0)
        src = src.clamp(min=0)

        def win(a):
            g = a[:, src]                             # [B,T,w,F]
            return g.reshape(Bsz * T, w, a.shape[-1])
        mask = valid[None].expand(Bsz, -1, -1).reshape(Bsz * T, w)
        y = selective_scan(win(u), win(delta), -torch.exp(self.A_log), win(Bm), win(Cm), self.D, mask)
        y = y[:, -1].reshape(Bsz, T, self.di)
        return self.out_proj(y * F.silu(z))
