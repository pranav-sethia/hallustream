"""End-to-end analysis for SHARS runs (EVAL_PROTOCOL.md, "SHARS with a probe detector").

Each arm is a directory with SHARS's bios.jsonl (from modal_jobs.shars run) and the biography
judge's results.jsonl. An empty biography is an abstention: no claims, not judged. Every arm
is compared with the greedy arm ("naive") when it is given, and with HalluSE when it is given.

Run: uv run python -m hallustream.research.shars_test <dir> naive halluse probe_45 ...
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from hallustream.research.factcheck import judge
from hallustream.research.factcheck.validate import read_results
from hallustream.research.shars_log import parse

PRONOUNS = {"He", "She", "His", "Her", "They", "Their", "The", "I", "It", "This"}


def length_penalized(precision: float, n_claims: int, gamma: int = 10) -> float:
    """FActScore with the length penalty (Min et al. 2023): precision * exp(1 - gamma / n)."""
    if n_claims == 0:
        return 0.0
    return precision * (math.exp(1 - gamma / n_claims) if n_claims < gamma else 1.0)


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


def specific(claim: str, name: str) -> bool:
    """A claim with a number or a capitalized word (past the first word) that is not part of
    the person's name, e.g. a place, title, date or organization. A rough split of concrete
    facts from generic ones ("has been active in the entertainment industry")."""
    own = set(re.findall(r"\w+", name.lower()))
    words = re.findall(r"[A-Za-z][\w'-]*|\d+", claim)
    return any(w.isdigit() or (i > 0 and w[0].isupper() and w.lower() not in own
                               and w not in PRONOUNS) for i, w in enumerate(words))  # fmt: skip


def specific_counts(d: Path) -> pd.DataFrame:
    """Specific supported / unsupported claims per person (0 for abstentions)."""
    results, _ = read_results(d / "results.jsonl")
    rows = []
    for line in (d / "bios.jsonl").read_text().splitlines():
        name = json.loads(line)["name"]
        claims = judge.parse_bio(results[name])["claims"] if name in results else []
        rows.append({"name": name,
                     "specific_supported": sum(lab == "Supported" and specific(t, name)
                                               for t, lab in claims),
                     "specific_unsupported": sum(lab != "Supported" and specific(t, name)
                                                 for t, lab in claims)})  # fmt: skip
    return pd.DataFrame(rows)


def arm_table(d: Path) -> pd.DataFrame:
    """Per-person metrics, with abstentions and timing."""
    bios = pd.DataFrame([json.loads(x) for x in (d / "bios.jsonl").read_text().splitlines()])
    judged = per_bio(d / "bios.jsonl", d / "results.jsonl")
    df = bios.merge(judged, on="name", how="left")
    empty = df["bio"].str.strip() == ""
    df.loc[empty, ["claims", "supported", "unsupported", "fs_gamma10"]] = 0
    df.loc[empty, "responded"] = False
    df["responded"] = df["responded"].astype("boolean")
    missing = df["responded"].isna()
    if missing.any():
        print(f"{d.name}: {int(missing.sum())} answered biographies without a judge result")
    df = df[~missing].copy()
    df["responded"] = df["responded"].astype(bool)
    df = df.merge(specific_counts(d), on="name", how="left")
    for c in ("detect_seconds", "decompose_seconds"):
        if c not in df:
            df[c] = 0.0
    return df


def log_stats(df: pd.DataFrame) -> dict:
    """What SHARS did with the sentences it tried, summed over people."""
    if "log" not in df or df["log"].isna().all():
        return {}
    counts: Counter = Counter()
    ends: Counter = Counter()
    rewrites = []
    for log in df["log"].fillna(""):
        p = parse(log)
        counts.update(p["counts"])
        ends[p["end"]] += 1
        rewrites += p["rewrites"]
    out = {"per_person": {k: round(v / len(df), 2) for k, v in sorted(counts.items())},
           "ends": dict(ends), "rewrites": len(rewrites)}  # fmt: skip
    if rewrites:
        n = len(rewrites)
        out["rewrites_keeping_a_flagged_entity"] = round(sum(bool(w["flagged_kept"]) for w in rewrites) / n, 3)  # fmt: skip  # noqa: E501
        out["rewrites_missing_a_trusted_entity"] = round(sum(bool(w["trusted_missing"]) for w in rewrites) / n, 3)  # fmt: skip  # noqa: E501
    if "props" in df and df["props"].notna().any():
        out["unlocated_share"] = round(float(df["unlocated"].sum() / max(df["props"].sum(), 1)), 4)
    return out


def report(arms: dict[str, pd.DataFrame]) -> dict:
    out = {"arms": {}}
    for k, df in arms.items():
        out["arms"][k] = summary(df) | {
            "precision_pooled": round(float(df["supported"].sum() / max(df["claims"].sum(), 1)), 4),
            "seconds_per_bio": round(float(df["seconds"].mean()), 1),
            "detect_seconds_per_bio": round(float(df["detect_seconds"].mean()), 2),
            "specific_supported_per_bio": round(float(df["specific_supported"].mean()), 2),
            "specific_unsupported_per_bio": round(float(df["specific_unsupported"].mean()), 2),
            "aborted_on_error": int(df.get("aborted_on_error", pd.Series(0)).sum()),
            "log": log_stats(df),
        }  # fmt: skip
        props = out["arms"][k]["log"].get("per_person", {}).get("props")
        if props:
            out["arms"][k]["detect_seconds_per_prop"] = round(
                float(df["detect_seconds"].mean()) / props, 4
            )
    pairs = [(a, b) for b in ("naive", "halluse") if b in arms for a in arms if a != b]
    pairs += [
        (a, b) for a in arms for b in arms if a.startswith("probe") and b.startswith("logprob")
    ]
    if "probe_45" in arms and "probe_v1" in arms:  # the adapted probe vs the paper's recipe
        pairs.append(("probe_45", "probe_v1"))
    out["paired"] = {f"{a}-{b}": {c: paired(arms[a], arms[b], c)
                                  for c in ("precision", "supported", "unsupported",
                                            "specific_supported", "specific_unsupported")}
                     for a, b in pairs}  # fmt: skip
    return out


def _json_default(x):
    return x.item() if hasattr(x, "item") else str(x)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    arms = {k: arm_table(root / k) for k in sys.argv[2:]}
    print(json.dumps(report(arms), indent=1, default=_json_default))
