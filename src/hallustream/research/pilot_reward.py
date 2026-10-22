"""Probe-reward pilot analysis (EVAL_PROTOCOL.md, "Probe-reward pilot").

Run: uv run python -m hallustream.research.pilot_reward <dir with bios.jsonl and results.jsonl>
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hallustream.research.factcheck import judge
from hallustream.research.factcheck.validate import read_results

CONFIDENCE = ["r_surprise", "r_entropy", "m_surprise", "m_entropy"]


def reward(p) -> float:
    """F / (T + 1) + 0.5 log(1 + F): T entities, F = expected number of correct ones."""
    p = np.asarray(p, dtype=float)
    f = float((1 - p).sum())
    return f / (len(p) + 1) + 0.5 * float(np.log1p(f))


def table(d: Path) -> pd.DataFrame:
    rows = [json.loads(x) for x in (d / "bios.jsonl").read_text().splitlines()]
    results, _ = read_results(d / "results.jsonl")
    for sig in ("surprise", "entropy"):  # confidence mapped to [0, 1] by rank over the pilot
        ref = np.sort(np.concatenate([r[sig] for r in rows if r[sig]]))
        for r in rows:
            r[f"{sig}_p"] = (np.searchsorted(ref, r[sig]) / len(ref)).tolist()
    recs = []
    for r in rows:
        try:
            claims = judge.parse_bio(results[r["key"]])["claims"]
        except (KeyError, ValueError):
            continue
        n = len(claims)
        sup = sum(label == "Supported" for _, label in claims)
        recs.append({
            "name": r["name"], "key": r["key"], "claims": n, "supported": sup,
            "unsupported": n - sup, "precision": sup / n if n else np.nan,
            "r_probe": reward(r["probe_p"]), "r_surprise": reward(r["surprise_p"]),
            "r_entropy": reward(r["entropy_p"]),
            "m_probe": -float(np.mean(r["probe_p"])) if r["probe_p"] else 0.0,
            "m_surprise": -float(np.mean(r["surprise"])) if r["surprise"] else 0.0,
            "m_entropy": -float(np.mean(r["entropy"])) if r["entropy"] else 0.0,
        })  # fmt: skip
    return pd.DataFrame(recs)


def analyze(df: pd.DataFrame, gap: float = 0.2) -> dict:
    answered = df[df["claims"] > 0]
    g = answered.groupby("name")["precision"].agg(["max", "min", "size"])
    g = g[g["size"] >= 2]
    signals = ["r_probe", "m_probe", *CONFIDENCE]
    hits: dict[str, list[float]] = {s: [] for s in signals}
    for _, grp in answered.groupby("name"):
        for a, b in itertools.combinations(grp.to_dict(orient="records"), 2):  # pyright: ignore[reportCallIssue]
            diff = a["precision"] - b["precision"]
            if abs(diff) < gap:
                continue
            for s in signals:
                d = a[s] - b[s]
                hits[s].append(0.5 if d == 0 else float(np.sign(d) == np.sign(diff)))
    groups = [grp for _, grp in df.groupby("name")]
    best = pd.DataFrame([grp.loc[grp["r_probe"].idxmax()] for grp in groups])
    rand = df.groupby("name")[["supported", "unsupported"]].mean()
    rand_precision = answered.groupby("name")["precision"].mean()
    return {
        "diversity": round(float(((g["max"] - g["min"]) >= gap).mean()), 3),
        "pairs": len(hits["r_probe"]),
        "pair_accuracy": {s: round(float(np.mean(v)), 3) for s, v in hits.items()},
        "best_of_n_precision": round(float(best["precision"].mean()), 3),
        "random_precision": round(float(rand_precision.mean()), 3),
        "best_of_n_supported": round(float(best["supported"].mean()), 2),
        "random_supported": round(float(rand["supported"].mean()), 2),
        "best_of_n_unsupported": round(float(best["unsupported"].mean()), 2),
        "random_unsupported": round(float(rand["unsupported"].mean()), 2),
    }


if __name__ == "__main__":
    print(json.dumps(analyze(table(Path(sys.argv[1]))), indent=1))
