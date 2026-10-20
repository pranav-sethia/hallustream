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
