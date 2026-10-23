"""Dev analysis for SHARS with a probe detector (EVAL_PROTOCOL.md, "SHARS with a probe detector").

Inputs: props_all.parquet (every dev proposition with every detector's score) and the claim
judge's results. Picks HalluSE's best setting, the probe's context, and matched thresholds.

Run: uv run python -m hallustream.research.shars_dev <props_all.parquet> <judge results.jsonl>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hallustream.research.factcheck import judge
from hallustream.research.factcheck.validate import read_results
from hallustream.research.metrics import auroc, paired_bootstrap


def labeled(props: pd.DataFrame, results: dict[str, str]) -> pd.DataFrame:
    """Attach the judge's label to every proposition (claims numbered in table order)."""
    parts = []
    for name, g in props.groupby("name", sort=True):
        try:
            labels = judge.parse_ids(results[str(name)], len(g))
        except (KeyError, ValueError):
            continue
        parts.append(g.assign(hallucinated=[lab == "Not Supported" for lab in labels]))
    return pd.concat(parts, ignore_index=True)


def youden(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold (flag if score > t) maximizing TPR - FPR, over the score's own values."""
    best, best_j = float("inf"), -1.0
    for t in np.unique(s):
        flag = s > t
        j = flag[y == 1].mean() - flag[y == 0].mean()
        if j > best_j:
            best, best_j = float(t), float(j)
    return best


def matched(s: np.ndarray, rate: float) -> float:
    """Threshold that flags `rate` of the propositions (flag if score > t)."""
    return float(np.quantile(s, 1 - rate))


def analyze(df: pd.DataFrame) -> dict:
    n_all = len(df)
    cols = [
        "halluse_a3",
        "halluse_a5",
        "native_v3",
        "shars_v3",
        "shars_surprise",
        "native_surprise",
    ]
    df = df.dropna(subset=cols)  # HalluSE errors (no question generated) or unplaced entities
    y = df["hallucinated"].to_numpy().astype(int)
    groups = df["name"].to_numpy()
    sig = lambda c: 1 / (1 + np.exp(-df[c].to_numpy()))  # noqa: E731
    # HalluSE variants: plain entropy; with the claimed entity as an extra answer ("exp");
    # and SHARS's entailment check, which also flags a proposition whose entity is not in
    # the largest answer cluster (encoded as +10, above any entropy threshold).
    halluse = {}
    for n in (3, 5):
        halluse[f"halluse_a{n}"] = df[f"halluse_a{n}"].to_numpy()
        if f"exp_a{n}" in df:
            miss = 10 * (1 - df[f"entailed_a{n}"].to_numpy())
            halluse[f"halluse_exp_a{n}"] = df[f"exp_a{n}"].to_numpy()
            halluse[f"halluse_entail_a{n}"] = df[f"halluse_a{n}"].to_numpy() + miss
            halluse[f"halluse_exp_entail_a{n}"] = df[f"exp_a{n}"].to_numpy() + miss
    scores = {
        **halluse,
        "probe_native": sig("native_v3"), "probe_shars": sig("shars_v3"),
        "surprise_shars": df["shars_surprise"].to_numpy(),
        "surprise_native": df["native_surprise"].to_numpy(),
    }  # fmt: skip
    res = {"people": int(df["name"].nunique()), "props": int(len(y)), "dropped": n_all - len(y),
           "hallucinated_share": round(float(y.mean()), 3),
           "auroc": {k: round(auroc(y, v), 4) for k, v in scores.items()}}  # fmt: skip
    h = max(halluse, key=lambda k: res["auroc"][k])
    p = max(("probe_native", "probe_shars"), key=lambda k: res["auroc"][k])
    s = max(("surprise_shars", "surprise_native"), key=lambda k: res["auroc"][k])
    th_h = youden(y, scores[h])
    rate = float((scores[h] > th_h).mean())
    choice = {
        "halluse": {"setting": h, "threshold": th_h, "flag_rate": round(rate, 3)},
        "probe_matched": {"setting": p, "threshold": matched(scores[p], rate)},
        "probe_best": {"setting": p, "threshold": youden(y, scores[p])},
        "surprise_matched": {"setting": s, "threshold": matched(scores[s], rate)},
    }
    for c in choice.values():
        v = scores[c["setting"]]
        flag = v > c["threshold"]
        c["flag_rate"] = round(float(flag.mean()), 3)
        c["tpr"] = round(float(flag[y == 1].mean()), 3)
        c["fpr"] = round(float(flag[y == 0].mean()), 3)
        c["kept_precision"] = round(float(1 - y[~flag].mean()), 3) if (~flag).any() else None
    res["choice"] = choice
    res["probe_minus_halluse"] = {
        k: round(v, 4) for k, v in paired_bootstrap(y, scores[p], scores[h], groups).items()
    }
    return res


if __name__ == "__main__":
    props = pd.read_parquet(sys.argv[1])
    results, _ = read_results(Path(sys.argv[2]))
    print(json.dumps(analyze(labeled(props, results)), indent=1))
