import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def tracing_metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    return dict(AUC=float(auc), ACC=float(((p >= 0.5) == (y >= 0.5)).mean()),
                RMSE=float(np.sqrt(((p - y) ** 2).mean())))


@torch.no_grad()
def ranking_hits(q, E_items, target, k=10, chunk=4096):
    """q [M,d], E_items [n,d], target [M]; ranks the true next item among ALL items. Returns (hit, ndcg) arrays."""
    hits, ndcg = [], []
    for s in range(0, len(q), chunk):
        sc = q[s:s + chunk] @ E_items.T
        t = target[s:s + chunk]
        rank = (sc > sc.gather(1, t[:, None])).sum(1)
        hits.append((rank < k).float().cpu())
        ndcg.append(torch.where(rank < k, 1.0 / torch.log2(rank.float() + 2), torch.zeros_like(rank, dtype=torch.float)).cpu())
    return torch.cat(hits).numpy(), torch.cat(ndcg).numpy()
