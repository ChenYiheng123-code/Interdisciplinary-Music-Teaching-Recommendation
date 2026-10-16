import copy
import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _cast(v: str):
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def load_config(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = Path(path).parent / "default.yaml"
    if Path(path).name != "default.yaml" and base.exists():
        with open(base) as f:
            merged = yaml.safe_load(f)
        _deep_update(merged, cfg)
        cfg = merged
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = _cast(v)
    return cfg


def _deep_update(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = copy.deepcopy(v)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def env_info():
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        info["git"] = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        info["git"] = None
    return info


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
