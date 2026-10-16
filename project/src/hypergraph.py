
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_incidence(mod_feats, clists, n_items, n_concepts, k, use_concepts=True):
    rows, cols = [], []
    off = 0
    for m, feat in mod_feats.items():
        f = F.normalize(torch.from_numpy(feat), dim=-1)
        for s in range(0, n_items, 2048):
            sim = f[s:s + 2048] @ f.T                     # centroid rows x all items
            kk = min(k, n_items)
            idx = sim.topk(kk, dim=-1).indices            # members of hyperedge (s + r)
            r = idx.reshape(-1)
            c = (torch.arange(s, min(s + 2048, n_items)).unsqueeze(1).expand(-1, kk).reshape(-1)) + off
            rows.append(r); cols.append(c)
        off += n_items
    if use_concepts:
        for i, cl in enumerate(clists):
            for c in cl:
                rows.append(torch.tensor([i])); cols.append(torch.tensor([off + c]))
        off += n_concepts
    if not rows:
        raise ValueError("empty hypergraph: no modalities and concept hyperedges disabled")
    rows = torch.cat(rows); cols = torch.cat(cols)
    H = torch.sparse_coo_tensor(torch.stack([rows, cols]), torch.ones(len(rows)), (n_items, off)).coalesce()
    H = torch.sparse_coo_tensor(H.indices(), torch.ones_like(H.values()), H.shape).coalesce()  # binary
    return H


def propagation_matrix(H, normalize=True):
    """Dense n x n operator (n <= ~20k)."""
    n, E = H.shape
    dv = torch.sparse.sum(H, 1).to_dense().clamp(min=1.0)
    de = torch.sparse.sum(H, 0).to_dense().clamp(min=1.0)
    if not normalize:
        return torch.sparse.mm(H, H.t().coalesce()).to_dense()
    idx = H.indices()
    Sv = torch.sparse_coo_tensor(idx, H.values() * dv[idx[0]].rsqrt(), H.shape).coalesce()
    SvW = torch.sparse_coo_tensor(idx, Sv.values() / de[idx[1]], H.shape).coalesce()  # W = I
    return torch.sparse.mm(SvW, Sv.t().coalesce()).to_dense()


class HypergraphEncoder(nn.Module):
    def __init__(self, n_items, d, mod_feats, clists, n_concepts, k=10, layers=3,
                 use_concepts=True, use_norm=True, use_residual=True, dropout=0.1):
        super().__init__()
        self.n, self.d = n_items, d
        self.mods = list(mod_feats.keys())
        self.use_residual = use_residual
        H = build_incidence(mod_feats, clists, n_items, n_concepts, k, use_concepts)
        self.register_buffer("A", propagation_matrix(H, use_norm))
        self.id_emb = nn.Embedding(n_items, d)
        nn.init.normal_(self.id_emb.weight, std=0.1)
        self.proj = nn.ModuleDict({m: nn.Linear(mod_feats[m].shape[1], d) for m in self.mods})   # Eq. 1
        self.gate = nn.ModuleDict({m: nn.Linear(2 * d, d) for m in self.mods})                    # Eq. 8
        for m in self.mods:
            self.register_buffer(f"feat_{m}", torch.from_numpy(mod_feats[m]))
        self.mod_w = nn.Parameter(torch.zeros(max(1, len(self.mods))))
        self.thetas = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

    def initial(self):
        ide = self.id_emb.weight
        if not self.mods:                       # symbolic-only data: gated denoiser == identity mapping
            return ide
        outs = []
        for m in self.mods:
            x = self.proj[m](getattr(self, f"feat_{m}"))
            g = torch.sigmoid(self.gate[m](torch.cat([x, ide], -1)))
            outs.append(x + g * x if self.use_residual else g * x)                                # Eq. 9
        w = torch.softmax(self.mod_w, 0)
        return sum(wi * o for wi, o in zip(w, outs))

    def forward(self):
        x = self.initial()
        for th in self.thetas:
            x = F.leaky_relu(self.A @ th(self.drop(x)), 0.2)
        return x
