"""Test analysis for SHARS with a probe detector (EVAL_PROTOCOL.md, "SHARS with a probe detector").

Each arm is a directory with SHARS's bios.jsonl (from modal_jobs.shars run) and the biography
judge's results.jsonl. An empty biography is an abstention: no claims, not judged.

Run: uv run python -m hallustream.research.shars_test <dir> naive halluse probe_matched ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from hallustream.fix.experiment import paired, per_bio, summary


def arm_table(d: Path) -> pd.DataFrame:
    """Per-person metrics, with abstentions and timing."""
    bios = pd.DataFrame([json.loads(x) for x in (d / "bios.jsonl").read_text().splitlines()])
    judged = per_bio(d / "bios.jsonl", d / "results.jsonl")
    df = bios.merge(judged, on="name", how="left")
    empty = df["bio"].str.strip() == ""
    df.loc[empty, ["responded", "claims", "supported", "unsupported", "fixes"]] = [
        False,
        0,
        0,
        0,
        0,
    ]
    df.loc[empty, "fs_gamma10"] = 0.0
    df["responded"] = df["responded"].astype("boolean")
    missing = df["responded"].isna()
    if missing.any():
        print(f"{d.name}: {int(missing.sum())} answered biographies without a judge result")
    df = df[~missing].copy()
    df["responded"] = df["responded"].astype(bool)
    for c in ("detect_seconds", "decompose_seconds"):
        if c not in df:
            df[c] = 0.0
    return df


def report(arms: dict[str, pd.DataFrame]) -> dict:
    out = {"arms": {}}
    for k, df in arms.items():
        out["arms"][k] = summary(df) | {
            "precision_pooled": round(float(df["supported"].sum() / max(df["claims"].sum(), 1)), 4),
            "seconds_per_bio": round(float(df["seconds"].mean()), 1),
            "detect_seconds_per_bio": round(float(df["detect_seconds"].mean()), 2),
            "aborted_on_error": int(df.get("aborted_on_error", pd.Series(0)).sum()),
        }  # fmt: skip
    cmp = {}
    pairs = [(a, b) for a in arms for b in arms if a != b and b in ("halluse", "naive")]
    pairs += [(a, "surprise_matched") for a in ("probe_matched", "probe_best") if a in arms]
    for a, b in pairs:
        if b not in arms:
            continue
        cmp[f"{a}-{b}"] = {c: paired(arms[a], arms[b], c)
                           for c in ("precision", "supported", "unsupported")}  # fmt: skip
    out["paired"] = cmp
    if {"probe_matched", "halluse"} <= arms.keys():
        p, h = out["arms"]["probe_matched"], out["arms"]["halluse"]
        s = out["arms"].get("surprise_matched")
        out["claim"] = {
            "precision_within_0.02": p["precision"] >= h["precision"] - 0.02,
            "supported_ratio": round(p["supported_per_bio"] / h["supported_per_bio"], 3),
            "supported_at_least_90pct": p["supported_per_bio"] >= 0.9 * h["supported_per_bio"],
            "detect_ratio": round(p["detect_seconds_per_bio"] / h["detect_seconds_per_bio"], 4),
            "detect_at_most_10pct": p["detect_seconds_per_bio"]
            <= 0.1 * h["detect_seconds_per_bio"],
            "beats_surprise": None if s is None else p["precision"] > s["precision"],
        }  # fmt: skip
    return out


if __name__ == "__main__":
    root = Path(sys.argv[1])
    arms = {k: arm_table(root / k) for k in sys.argv[2:]}
    print(
        json.dumps(
            report(arms), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)
        )
    )
