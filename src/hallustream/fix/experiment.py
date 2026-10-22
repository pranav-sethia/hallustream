"""Choose which draft sentences each trigger fixes, and score the judged results.

Run (module hallustream.fix.experiment):
  flags <sentences.parquet> <signal> <rate> <out.json> [threshold]   # threshold overrides rate
  random <sentences.parquet> <rate> <seed> <out.json>
  oracle <sentences.parquet> <draft bios.jsonl> <judge results.jsonl> <out.json>
  metrics <bios.jsonl> <judge results.jsonl> [<bios.jsonl> <results.jsonl> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hallustream.fix.engine import length_penalized
from hallustream.research.factcheck import judge
from hallustream.research.factcheck.validate import read_results

SIGNALS = ["v3_max", "v1_max", "probe_token_max", "surprise_max", "entropy_max"]


def _eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Only sentences with at least one entity can be flagged, whatever the trigger."""
    return df.loc[df["n_entities"] > 0]


def threshold_for_rate(df: pd.DataFrame, signal: str, rate: float) -> float:
    """Threshold at which the trigger flags `rate` sentences per biography on average."""
    s = np.sort(_eligible(df)[signal].to_numpy())[::-1]
    n = int(round(rate * df["name"].nunique()))
    if n <= 0:
        return float("inf")
    return float(s[min(n, len(s)) - 1])


def flags_at(df: pd.DataFrame, signal: str, threshold: float) -> dict:
    e = _eligible(df)
    hit = e[e[signal] >= threshold]
    return {n: sorted(g["sentence"].tolist()) for n, g in hit.groupby("name")}


def random_flags(df: pd.DataFrame, rate: float, seed: int = 0) -> dict:
    """Each eligible sentence flagged independently, at the given rate per biography."""
    e = _eligible(df)
    p = min(1.0, rate * df["name"].nunique() / max(len(e), 1))
    keep = np.random.default_rng(seed).random(len(e)) < p
    hit = e[keep]
    return {n: sorted(g["sentence"].tolist()) for n, g in hit.groupby("name")}


def sentence_labels(df: pd.DataFrame, bios: dict[str, str], results: dict[str, str]) -> pd.Series:
    """True where a draft sentence contains an entity the judge marked Not Supported."""
    bad = set()
    for name, g in df.groupby("name"):
        try:
            ents = judge.parse_bio(results[str(name)])["entities"]
        except (KeyError, ValueError):
            continue
        anns, _ = judge.locate(bios[str(name)], ents)
        for a in anns:
            if a["label"] != "Not Supported":
                continue
            for lo, hi, i in zip(g["start"], g["end"], g["sentence"], strict=True):
                if lo <= a["index"] < hi:
                    bad.add((name, i))
    return pd.Series([(n, int(i)) in bad for n, i in zip(df["name"], df["sentence"], strict=True)],
                     index=df.index)  # fmt: skip


def per_bio(bios_path: Path, results_path: Path) -> pd.DataFrame:
    """Claims-based metrics per biography from the judge's answers."""
    results, _ = read_results(results_path)
    rows = []
    for line in bios_path.read_text().splitlines():
        r = json.loads(line)
        try:
            claims = judge.parse_bio(results[r["name"]])["claims"]
        except (KeyError, ValueError):
            claims = None
        if claims is None:
            continue
        sup = sum(lab == "Supported" for _, lab in claims)
        n = len(claims)
        rows.append({
            "name": r["name"], "responded": n > 0, "claims": n, "supported": sup,
            "unsupported": n - sup, "precision": sup / n if n else np.nan,
            "fs_gamma10": length_penalized(sup / n, n) if n else 0.0,
            "fixes": len(r.get("flags", [])),
        })  # fmt: skip
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> dict:
    answered = df[df["responded"]]
    return {
        "bios": len(df), "response_rate": round(float(df["responded"].mean()), 4),
        "precision": round(float(answered["precision"].mean()), 4),
        "supported_per_bio": round(float(df["supported"].mean()), 2),
        "unsupported_per_bio": round(float(df["unsupported"].mean()), 2),
        "fs_gamma10": round(float(answered["fs_gamma10"].mean()), 4),
        "fixes_per_bio": round(float(df["fixes"].mean()), 3),
    }  # fmt: skip


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str, n_boot: int = 2000, seed: int = 0):
    """Mean difference a - b over people who answered in both, with a bootstrap interval."""
    m = a.merge(b, on="name", suffixes=("_a", "_b"))
    m = m[m["responded_a"] & m["responded_b"]] if col == "precision" else m
    d = np.asarray(m[f"{col}_a"], dtype=float) - np.asarray(m[f"{col}_b"], dtype=float)
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"delta": round(float(d.mean()), 4), "ci": [round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4)], "people": int(len(d))}  # fmt: skip


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "flags":
        df = pd.read_parquet(sys.argv[2])
        signal, rate = sys.argv[3], float(sys.argv[4])
        th = float(sys.argv[6]) if len(sys.argv) > 6 else threshold_for_rate(df, signal, rate)
        fl = flags_at(df, signal, th)
        Path(sys.argv[5]).write_text(json.dumps({"threshold": th, "flags": fl}))
        print(f"threshold {th:.4f}, {sum(map(len, fl.values())) / df['name'].nunique():.3f}/bio")
    elif cmd == "random":
        df = pd.read_parquet(sys.argv[2])
        fl = random_flags(df, float(sys.argv[3]), int(sys.argv[4]))
        Path(sys.argv[5]).write_text(json.dumps({"flags": fl}))
        print(f"{sum(map(len, fl.values())) / df['name'].nunique():.3f}/bio")
    elif cmd == "oracle":
        df = pd.read_parquet(sys.argv[2])
        bios = {
            (r := json.loads(x))["name"]: r["bio"]
            for x in Path(sys.argv[3]).read_text().splitlines()
        }
        df["bad"] = sentence_labels(df, bios, read_results(Path(sys.argv[4]))[0])
        e = _eligible(df)
        fl = {n: sorted(g["sentence"].tolist()) for n, g in e[e["bad"]].groupby("name")}
        Path(sys.argv[5]).write_text(json.dumps({"flags": fl}))
        print(f"{sum(map(len, fl.values())) / df['name'].nunique():.3f}/bio")
    elif cmd == "metrics":
        args = sys.argv[2:]
        for b, r in zip(args[::2], args[1::2], strict=True):
            print(Path(b).parent.name, summary(per_bio(Path(b), Path(r))))
