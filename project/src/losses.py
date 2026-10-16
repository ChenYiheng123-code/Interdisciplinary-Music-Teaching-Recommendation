import torch
import torch.nn.functional as F


def bce_loss(logit, y, mask):
    l = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    return (l * mask).sum() / mask.sum().clamp(min=1)


def bpr_loss(out, items, mask, n_items):
    """Positive = item actually interacted at step t, scored from the state before t; negative = random uniform item."""
    q, E = out["pref_q"], out["E"]
    pos = (q * E[items]).sum(-1)
    neg_items = torch.randint(0, n_items, items.shape, device=items.device)
    neg_items = torch.where(neg_items == items, (neg_items + 1) % n_items, neg_items)
    neg = (q * E[neg_items]).sum(-1)
    m = mask.clone(); m[:, 0] = False                                   # t=0 state is only the prototype
    l = -F.logsigmoid(pos - neg)
    return (l * m).sum() / m.sum().clamp(min=1)


def infonce_loss(H, id_emb, items, tau, max_items, n_items):
    u = torch.unique(items[items < n_items])
    if len(u) > max_items:
        u = u[torch.randperm(len(u), device=u.device)[:max_items]]
    z1 = F.normalize(H[u], dim=-1); z2 = F.normalize(id_emb[u], dim=-1)
    logits = z1 @ z2.T / tau
    return F.cross_entropy(logits, torch.arange(len(u), device=u.device))


def max_entropy_loss(gate_dense, mask, eps=1e-8):
    """Batch-level maximum-entropy regulariser: minimise sum p̄ log p̄ (= maximise entropy of mean expert activation)."""
    m = mask.unsqueeze(-1).float()
    pbar = (gate_dense * m).sum((0, 1)) / m.sum().clamp(min=1)
    return (pbar * torch.log(pbar + eps)).sum()
