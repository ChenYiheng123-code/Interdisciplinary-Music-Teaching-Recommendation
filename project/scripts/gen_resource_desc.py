
import argparse
from pathlib import Path

import pandas as pd

from musichg.llm_agent import QwenLLM, read_prompt
from musichg.utils import load_json

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--root", default="data/processed")
a = ap.parse_args()
lock = load_json("models.lock.json") if Path("models.lock.json").exists() else {}
llm = QwenLLM(revision=lock.get("llm", {}).get("revision"), max_new_tokens=160)
items = pd.read_csv(Path(a.root) / a.dataset / "items.csv")
tpl = read_prompt("offline_resource.txt")
descs = [llm.generate(tpl.format(Title=r.title, Attr=r.attrs if isinstance(r.attrs, str) else "")).strip() for r in items.itertuples()]
pd.DataFrame(dict(item=items["item"], desc=descs)).to_csv(Path(a.root) / a.dataset / "resource_desc.csv", index=False)
