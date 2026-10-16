
import numpy as np
import torch
import torch.nn.functional as F


def make_hist(items, fb, dt, dur, dev):
    t = lambda a, dtype: torch.as_tensor(np.asarray(a), dtype=dtype, device=dev)[None]
    h = dict(items=t(items, torch.long), fb=t(fb, torch.float), dt=t(dt, torch.float), dur=t(np.nan_to_num(dur, nan=30.0), torch.float))
    h["mask"] = torch.ones_like(h["items"], dtype=torch.bool)
    return h


def _append(h, items, fb, dt, dur):
    n = h["items"].shape[0]
    col = lambda v, dtype: torch.full((n, 1), v, dtype=dtype, device=h["items"].device) if not torch.is_tensor(v) else v.reshape(n, 1).to(dtype)
    return dict(items=torch.cat([h["items"], col(items, torch.long)], 1), fb=torch.cat([h["fb"], col(fb, torch.float)], 1),
                dt=torch.cat([h["dt"], col(dt, torch.float)], 1), dur=torch.cat([h["dur"], col(dur, torch.float)], 1),
                mask=torch.cat([h["mask"], torch.ones(n, 1, dtype=torch.bool, device=h["items"].device)], 1))


def _gain(m_end, m0, eps):
    # Eq. 5 (reconstructed): bounded normalised gain over target concepts
    return ((m_end - m0).sum(-1) / ((1 - m0).sum(-1) + eps)).clamp(0, 1)


@torch.no_grad()
def state_and_mastery(model, hist, targets):
    out = model(hist)
    S = out["S"][:, -1]
    return S, model.mastery_of(S, targets)


@torch.no_grad()
def simulate_path(model, hist, path, targets, eps=1e-6):
    """Roll a fixed path forward; returns (G_p, R_smooth)."""
    _, H = model.item_table()
    dt_med = hist["dt"].median().item(); dur_med = hist["dur"].median().item()
    _, m0 = state_and_mastery(model, hist, targets)
    h = hist
    for i in path:
        ext = _append(h, i, 0.0, dt_med, dur_med)
        p = torch.sigmoid(model(ext)["logit"][:, -1])
        h = _append(h, i, p, dt_med, dur_med)
    _, m1 = state_and_mastery(model, h, targets)
    G = _gain(m1, m0, eps).item()
    R = float(np.mean([(H[path[j]] - H[path[j + 1]]).norm().item() for j in range(len(path) - 1)])) if len(path) > 1 else 0.0
    return G, R


@torch.no_grad()
def beam_search(model, hist, cands, targets, B=3, L=5, lam=0.1, eps=1e-6):
    dev = hist["items"].device
    E, H = model.item_table()
    dt_med = hist["dt"].median().item(); dur_med = hist["dur"].median().item()
    cands = torch.as_tensor(cands, dtype=torch.long, device=dev)
    N = len(cands)
    _, m0 = state_and_mastery(model, hist, targets)
    beams = [dict(h=hist, path=[], dist=0.0, J=0.0, G=0.0)]
    for step in range(L):
        Bm = len(beams)
        Hall = {k: torch.cat([b["h"][k] for b in beams], 0).repeat_interleave(N, 0) for k in hist}
        cand_t = cands.repeat(Bm)
        ext = _append(Hall, cand_t, 0.0, dt_med, dur_med)
        p = torch.sigmoid(model(ext)["logit"][:, -1])
        ext2 = _append(Hall, cand_t, p, dt_med, dur_med)
        S2 = model(ext2)["S"][:, -1]
        G = _gain(model.mastery_of(S2, targets), m0, eps)                  # [Bm*N]
        J = torch.empty(Bm * N, device=dev)
        for bi, b in enumerate(beams):
            sl = slice(bi * N, (bi + 1) * N)
            if b["path"]:
                dist = b["dist"] + (H[cands] - H[b["path"][-1]]).norm(dim=-1)
                R = dist / len(b["path"])
            else:
                R = torch.zeros(N, device=dev)
            Jb = G[sl] - lam * R
            used = torch.tensor([c in b["path"] for c in cands.tolist()], device=dev)
            Jb = Jb.masked_fill(used, float("-inf"))
            J[sl] = Jb
        top = J.topk(min(B, Bm * N)).indices.tolist()
        new = []
        for idx in top:
            bi, ci = divmod(idx, N)
            b = beams[bi]; item = int(cands[ci])
            d = b["dist"] + ((H[item] - H[b["path"][-1]]).norm().item() if b["path"] else 0.0)
            new.append(dict(h={k: v[idx:idx + 1] for k, v in ext2.items()}, path=b["path"] + [item], dist=d, J=float(J[idx]), G=float(G[idx])))
        beams = new
    best = max(beams, key=lambda b: b["J"])
    R = best["dist"] / max(1, len(best["path"]) - 1)
    return best["path"], dict(G=best["G"], R=R, J=best["J"])


@torch.no_grad()
def greedy_path(model, hist, exclude, L=5):
    """Baseline protocol from the paper: top-L distinct items by predicted preference score."""
    out = model(hist)
    sc = out["pref_q"][:, -1] @ out["E"][: model.n].T
    sc = sc[0].clone()
    sc[list(exclude)] = float("-inf")
    return sc.topk(L).indices.tolist()


@torch.no_grad()
def candidate_set(model, hist, targets, desc, concept_init, clists, n_items, N, n_weak, exclude):
    """Phase-level plan (heuristic, no LLM) = target concepts U weakest concepts; candidates = top-N by cosine similarity
    between the plan vector and resource description vectors (Eq. 30).
    desc + concept_init (Sentence-BERT, same space) -> plan vector = mean concept embedding;
    otherwise concept-bag fallback (multi-hot)."""
    S = model(hist)["S"][:, -1]
    weak = model.mastery_all(S)[0].topk(n_weak, largest=False).indices.tolist()
    plan = sorted(set(targets.tolist()) | set(weak))
    dev = hist["items"].device
    if desc is not None and concept_init is not None:
        desc_t = torch.from_numpy(desc).to(dev)
        qv = torch.from_numpy(concept_init[plan].mean(0)).to(dev)
    else:
        V = np.zeros((n_items, model.nc), dtype=np.float32)
        for i, cl in enumerate(clists):
            for c in cl:
                V[i, c] = 1
        q = np.zeros(model.nc, dtype=np.float32); q[plan] = 1
        desc_t = torch.from_numpy(V).to(dev); qv = torch.from_numpy(q).to(dev)
    sim = F.cosine_similarity(desc_t, qv[None], dim=-1)
    sim[list(exclude)] = float("-inf")
    return sim.topk(min(N, n_items - len(exclude))).indices.tolist(), plan
