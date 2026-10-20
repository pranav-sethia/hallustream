"""Pick the probe layer from a sweep cache, using validation data only.

For each layer: fit an L2 logistic regression on the training rows (standardized
features, balanced classes), then score validation entities by their riskiest
token and compute span-level AUROC. The same simple linear method is used for
every layer, so the ranking is a fair comparison.

Run: uv run python -m hallustream.research.layer_sweep data/features/sweep_qwen35
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hallustream.research.metrics import auroc, recall_at_fpr


def load_run(run_dir: Path, layer: int) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    rows, feats, spans = [], [], []
    for shard in sorted(run_dir.glob("shard*")):
        for sp in sorted(shard.glob("spans_c*.parquet")):
            c = sp.stem.split("_c")[1]
            rows.append(pd.read_parquet(shard / f"rows_c{c}.parquet"))
            feats.append(np.load(shard / f"feat_L{layer}_c{c}.npy"))
            spans.append(pd.read_parquet(sp))
    return pd.concat(rows, ignore_index=True), np.concatenate(feats), pd.concat(spans)


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


def evaluate_layer(run_dir: Path, layer: int, C: float = 1.0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rows, X, spans = load_run(run_dir, layer)
    tr, va = (rows["split"] == "train").to_numpy(), (rows["split"] == "val").to_numpy()
    scaler = StandardScaler().fit(X[tr].astype(np.float32))
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(X[tr].astype(np.float32)), rows["label"].to_numpy()[tr])
    p = clf.predict_proba(scaler.transform(X[va].astype(np.float32)))[:, 1]
    vs = spans[spans["split"] == "val"]
    scores = span_max_scores(rows[va], p, vs)
    y = vs["label"].to_numpy()
    return {
        "layer": layer,
        "val_auroc": round(auroc(y, scores), 4),
        "val_recall_at_10pct_fpr": round(recall_at_fpr(y, scores), 4),
        "n_train_rows": int(tr.sum()),
        "n_val_spans": int(len(vs)),
    }


def baselines(run_dir: Path, layer: int) -> dict:
    _, _, spans = load_run(run_dir, layer)
    vs = spans[spans["split"] == "val"]
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
        "val_baselines": baselines(run, layers[0]),
    }
    res["best_layer"] = max(res["layers"], key=lambda r: r["val_auroc"])["layer"]
    return res


if __name__ == "__main__":
    out = main(sys.argv[1])
    print(json.dumps(out, indent=1))
    Path(sys.argv[1], "sweep_results.json").write_text(json.dumps(out, indent=1))
