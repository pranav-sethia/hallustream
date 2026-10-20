"""Span-level metrics, matching Obeso et al.'s evaluation."""

from __future__ import annotations

import numpy as np

from hallustream.research.align import Span


def span_scores(probs: np.ndarray, spans: list[Span]) -> tuple[np.ndarray, np.ndarray]:
    """Score each span by its riskiest token (inclusive bounds)."""
    scores = np.array([probs[s.start : s.end + 1].max() for s in spans], dtype=np.float64)
    labels = np.array([s.label for s in spans], dtype=np.int64)
    return scores, labels


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def recall_at_fpr(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(fpr <= max_fpr)[0]
    return float(tpr[idx[-1]]) if len(idx) else 0.0


def paired_bootstrap(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """AUROC(a) - AUROC(b) with a 95% interval, resampling whole groups (questions).

    Spans from the same question are correlated, so resampling spans one by one
    would make the interval too narrow.
    """
    rng = np.random.default_rng(seed)
    _, inverse = np.unique(groups, return_inverse=True)
    members = np.split(np.argsort(inverse, kind="stable"), np.cumsum(np.bincount(inverse))[:-1])
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        deltas[i] = auroc(labels[idx], scores_a[idx]) - auroc(labels[idx], scores_b[idx])
    return {
        "delta": auroc(labels, scores_a) - auroc(labels, scores_b),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "frac_le_0": float((deltas <= 0).mean()),
    }
