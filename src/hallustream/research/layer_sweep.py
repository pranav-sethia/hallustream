"""Pick the probe layer (or layer combination) from a sweep cache, using validation data only.

For each candidate: fit an L2 logistic regression on the training rows (standardized
features, balanced classes), then score validation entities by their riskiest
token and compute span-level AUROC. The same simple linear method is used for
every candidate, so the ranking is a fair comparison.

A candidate is one layer, several layers side by side ("concat"), or the average
of several layers ("mean", same size as a single layer).

Run: uv run python -m hallustream.research.layer_sweep data/features/sweep_qwen35
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hallustream.research.metrics import auroc, recall_at_fpr


def _chunks(run_dir: Path) -> list[tuple[Path, str]]:
    return [
        (shard, sp.stem.split("_c")[1])
        for shard in sorted(run_dir.glob("shard*"))
        for sp in sorted(shard.glob("spans_c*.parquet"))
    ]


def load_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunks = _chunks(run_dir)
    rows = pd.concat([pd.read_parquet(s / f"rows_c{c}.parquet") for s, c in chunks])
    spans = pd.concat([pd.read_parquet(s / f"spans_c{c}.parquet") for s, c in chunks])
    return rows.reset_index(drop=True), spans.reset_index(drop=True)


def load_features(run_dir: Path, layers: list[int], combine: str = "concat") -> np.ndarray:
    """Features for every row, in load_tables order. float16 to keep memory down."""
    out = []
    for s, c in _chunks(run_dir):
        parts = [np.load(s / f"feat_L{layer}_c{c}.npy") for layer in layers]
        if combine == "mean":
            out.append(np.mean([p.astype(np.float32) for p in parts], axis=0).astype(np.float16))
        else:
            out.append(np.hstack(parts))
    return np.concatenate(out)


def load_run(run_dir: Path, layer: int) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    rows, spans = load_tables(run_dir)
    return rows, load_features(run_dir, [layer]), spans


def span_max_scores(rows: pd.DataFrame, probs: np.ndarray, spans: pd.DataFrame) -> np.ndarray:
    """Max token probability inside each span (token ranges are inclusive)."""
    by_answer: dict[str, dict[int, float]] = {}
    for aid, pos, p in zip(rows["answer_id"], rows["token_pos"], probs, strict=True):
        by_answer.setdefault(aid, {})[pos] = p
    out = []
    for aid, s, e in zip(spans["answer_id"], spans["start"], spans["end"], strict=True):
        tok = by_answer[aid]
        out.append(max(tok[t] for t in range(s, e + 1)))
    return np.array(out)


def fit_and_score(
    rows: pd.DataFrame,
    X: np.ndarray,
    spans: pd.DataFrame,
    exclude: frozenset[str] = frozenset(),
    C: float = 1.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Train on train rows, return span-max scores for the validation spans."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    keep = ~rows["answer_id"].isin(exclude).to_numpy()
    tr = (rows["split"] == "train").to_numpy() & keep
    va = (rows["split"] == "val").to_numpy() & keep
    Xtr = X[tr].astype(np.float32)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), rows["label"].to_numpy()[tr])
    del Xtr
    p = clf.predict_proba(scaler.transform(X[va].astype(np.float32)))[:, 1]
    vs = spans[(spans["split"] == "val") & ~spans["answer_id"].isin(exclude)]
    return span_max_scores(rows[va], p, vs), vs


def evaluate(
    run_dir: Path, layers: list[int], combine: str = "concat", exclude=frozenset()
) -> dict:
    rows, spans = load_tables(run_dir)
    scores, vs = fit_and_score(rows, load_features(run_dir, layers, combine), spans, exclude)
    y = vs["label"].to_numpy()
    return {
        "layers": layers,
        "combine": combine,
        "val_auroc": round(auroc(y, scores), 4),
        "val_recall_at_10pct_fpr": round(recall_at_fpr(y, scores), 4),
        "n_val_spans": int(len(vs)),
        "scores": scores.round(5).tolist(),  # for paired bootstraps between candidates
        "span_answer_ids": vs["answer_id"].tolist(),
        "labels": y.tolist(),
    }


def evaluate_layer(run_dir: Path, layer: int, C: float = 1.0) -> dict:
    r = evaluate(run_dir, [layer])
    return {"layer": layer, **{k: r[k] for k in ("val_auroc", "val_recall_at_10pct_fpr")}}


def baselines(run_dir: Path, exclude=frozenset()) -> dict:
    _, spans = load_tables(run_dir)
    vs = spans[(spans["split"] == "val") & ~spans["answer_id"].isin(exclude)]
    y = vs["label"].to_numpy()
    return {
        c: round(auroc(y, vs[c].to_numpy()), 4)
        for c in ["nll_max", "nll_mean", "entropy_max", "entropy_mean"]
    }


def main(run_dir: str) -> dict:
    run = Path(run_dir)
    meta = json.loads(next(run.glob("shard*/run_meta.json")).read_text())
    layers = meta["config"]["layers"]
    res = {
        "model": meta["config"]["model"],
        "layers": [evaluate_layer(run, layer) for layer in layers],
        "val_baselines": baselines(run),
    }
    res["best_layer"] = max(res["layers"], key=lambda r: r["val_auroc"])["layer"]
    return res


if __name__ == "__main__":
    out = main(sys.argv[1])
    print(json.dumps(out, indent=1))
    Path(sys.argv[1], "sweep_results.json").write_text(json.dumps(out, indent=1))
