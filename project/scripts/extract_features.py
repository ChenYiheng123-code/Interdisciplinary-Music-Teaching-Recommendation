
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from musichg.utils import load_json


def rev(lock, k):
    return lock.get(k, {}).get("revision")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--media", required=True)
    ap.add_argument("--root", default="data/processed"); ap.add_argument("--out", default="data/features")
    ap.add_argument("--models", default="models.yaml"); ap.add_argument("--lock", default="models.lock.json")
    ap.add_argument("--only", nargs="*", default=["vit", "audio", "text", "sbert"])
    a = ap.parse_args()
    M = yaml.safe_load(open(a.models)); lock = load_json(a.lock) if Path(a.lock).exists() else {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = pd.read_csv(Path(a.root) / a.dataset / "items.csv"); conc = pd.read_csv(Path(a.root) / a.dataset / "concepts.csv")
    out = Path(a.out) / a.dataset; out.mkdir(parents=True, exist_ok=True)
    media = Path(a.media)

    def run(name, fn, dim):
        feats = np.zeros((len(items), dim), dtype=np.float32)
        miss = 0
        for i, it in enumerate(items["item"].astype(str)):
            v = fn(media / it)
            if v is None: miss += 1
            else: feats[i] = v
        if miss: print(f"[warn] {name}: {miss}/{len(items)} items missing -> zeros")
        np.save(out / f"{name}.npy", feats)

    with torch.no_grad():
        if "vit" in a.only:
            from PIL import Image
            from transformers import ViTImageProcessor, ViTModel
            n = M["image_encoder"]["name"]; proc = ViTImageProcessor.from_pretrained(n, revision=rev(lock, "image_encoder"))
            vit = ViTModel.from_pretrained(n, revision=rev(lock, "image_encoder")).to(dev).eval()
            def f(p):
                p = p / "keyframe.jpg"
                if not p.exists(): return None
                x = proc(images=Image.open(p).convert("RGB"), return_tensors="pt").to(dev)
                return vit(**x).last_hidden_state[:, 0].squeeze(0).cpu().numpy()
            run("vit", f, vit.config.hidden_size)
        if "audio" in a.only:
            import librosa
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
            n = M["audio_encoder"]["name"]
            fe = Wav2Vec2FeatureExtractor.from_pretrained(n, revision=rev(lock, "audio_encoder"), trust_remote_code=True)
            am = AutoModel.from_pretrained(n, revision=rev(lock, "audio_encoder"), trust_remote_code=True).to(dev).eval()
            def f(p):
                p = p / "audio.wav"
                if not p.exists(): return None
                y, _ = librosa.load(p, sr=fe.sampling_rate, mono=True, duration=30)
                x = fe(y, sampling_rate=fe.sampling_rate, return_tensors="pt").to(dev)
                return am(**x).last_hidden_state.mean(1).squeeze(0).cpu().numpy()
            run("audio", f, am.config.hidden_size)
        if "text" in a.only:
            from transformers import AutoModel, AutoTokenizer
            n = M["text_encoder"]["name"]; tk = AutoTokenizer.from_pretrained(n, revision=rev(lock, "text_encoder"))
            tm = AutoModel.from_pretrained(n, revision=rev(lock, "text_encoder")).to(dev).eval()
            def f(p):
                p = p / "text.txt"
                if not p.exists(): return None
                x = tk(p.read_text(encoding="utf-8")[:4000], truncation=True, max_length=512, return_tensors="pt").to(dev)
                return tm(**x).last_hidden_state[:, 0].squeeze(0).cpu().numpy()
            run("text", f, tm.config.hidden_size)
        if "sbert" in a.only:
            from sentence_transformers import SentenceTransformer
            n = M["sentence_encoder"]["name"]
            sb = SentenceTransformer(n, revision=rev(lock, "sentence_encoder"), device=dev)
            np.save(out / "concept_init.npy", sb.encode(conc["name"].astype(str).tolist(), normalize_embeddings=True).astype(np.float32))
            dfile = Path(a.root) / a.dataset / "resource_desc.csv"   # produced by scripts/gen_resource_desc.py (offline LLM)
            txt = pd.read_csv(dfile)["desc"].astype(str).tolist() if dfile.exists() else (items["title"].astype(str) + " " + items["attrs"].fillna("").astype(str)).tolist()
            if not dfile.exists(): print("[warn] resource_desc.csv missing -> using title+attrs instead of LLM-generated functional text")
            np.save(out / "desc.npy", sb.encode(txt, normalize_embeddings=True).astype(np.float32))


if __name__ == "__main__":
    main()
