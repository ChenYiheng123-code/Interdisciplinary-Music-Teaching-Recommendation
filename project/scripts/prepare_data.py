
import argparse
from pathlib import Path

import pandas as pd


def assist2017(raw, out):
    df = pd.read_csv(raw, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    pick = lambda *names: next(cols[n.lower()] for n in names if n.lower() in cols)
    m = dict(user=pick("studentId"), item=pick("problemId"), skill=pick("skill"), correct=pick("correct"),
             ts=pick("startTime"), dur=pick("timeTaken"))
    print("column mapping:", m)
    df = df[[m[k] for k in ("user", "item", "skill", "correct", "ts", "dur")]].dropna(subset=[m["skill"], m["item"]])
    df.columns = ["user", "item", "skill", "correct", "ts", "duration"]
    df["timestamp"] = pd.to_numeric(df["ts"], errors="coerce")
    if df["timestamp"].isna().mean() > 0.5:
        df["timestamp"] = pd.to_datetime(df["ts"], errors="coerce").astype("int64") // 10**9
    df = df.dropna(subset=["timestamp"])
    df["correct"] = (df["correct"] > 0.5).astype(int)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df[["user", "item", "correct", "timestamp", "duration"]].to_csv(out / "interactions.csv", index=False)
    it = df.groupby("item")["skill"].agg(lambda s: "|".join(sorted(set(map(str, s))))).reset_index()
    it["title"] = it["item"].astype(str); it["attrs"] = ""; it = it.rename(columns={"skill": "concepts"})
    it[["item", "title", "attrs", "concepts"]].to_csv(out / "items.csv", index=False)
    sk = sorted(set(df["skill"].astype(str)))
    pd.DataFrame(dict(concept=sk, name=sk)).to_csv(out / "concepts.csv", index=False)
    print(f"assist2017: {len(df)} rows, {df['user'].nunique()} users, {df['item'].nunique()} items, {len(sk)} skills")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["assist2017", "moocccubex", "musicnet"])
    ap.add_argument("--raw"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.dataset == "assist2017":
        assist2017(a.raw, a.out or "data/processed/assist2017")
    else:
        raise SystemExit(__doc__)
