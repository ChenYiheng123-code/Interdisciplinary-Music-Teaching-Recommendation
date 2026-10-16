import numpy as np
import torch

from musichg.ssm import MambaBlock


def test_backward_window_is_causal():

    torch.manual_seed(0)
    blk = MambaBlock(16, 4).eval()
    x = torch.randn(2, 12, 16)
    y1 = blk.forward_window(x, 5)
    x2 = x.clone(); x2[:, 7:] += torch.randn(2, 5, 16)
    y2 = blk.forward_window(x2, 5)
    assert torch.allclose(y1[:, :7], y2[:, :7], atol=1e-5)
    assert not torch.allclose(y1[:, 8:], y2[:, 8:], atol=1e-5)


def test_forward_scan_is_causal():
    torch.manual_seed(0)
    blk = MambaBlock(16, 4).eval()
    x = torch.randn(2, 12, 16)
    y1 = blk(x)
    x2 = x.clone(); x2[:, 9:] += 1.0
    assert torch.allclose(y1[:, :9], blk(x2)[:, :9], atol=1e-5)


def test_splits_no_leakage(tmp_path):
    from musichg.data import make_splits
    rng = np.random.default_rng(0)
    sess = []
    for i in range(200):
        st = rng.uniform(0, 1e6)
        sess.append(dict(sid=f"s{i}", ts=np.sort(st + rng.uniform(0, 5e4, 6))))
    sp = make_splits(sess)
    by = {s["sid"]: s for s in sess}
    t1, t2 = sp["cutoffs"]["t_train_end"], sp["cutoffs"]["t_val_end"]
    assert all(by[i]["ts"][-1] <= t1 for i in sp["train"])
    assert all(t1 <= by[i]["ts"][0] and by[i]["ts"][-1] <= t2 for i in sp["val"])
    assert all(by[i]["ts"][0] >= t2 for i in sp["test"])
