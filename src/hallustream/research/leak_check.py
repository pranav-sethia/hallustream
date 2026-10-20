"""Look for reworded copies of test questions in train/val.

The split removes exact (normalized) question matches. This catches near-duplicates
it can miss: for each test question, find its most similar train/val question by
TF-IDF cosine similarity and report how many pairs pass a few cutoffs. Train/val
answers to questions at or above EXCLUDE_AT are listed for removal before training.

Run: uv run python -m hallustream.research.leak_check
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hallustream.research.data import load_records, normalize_question

CUTOFFS = (0.7, 0.8, 0.9)
# Read by hand: every pair at >= 0.9 asks about the same subject (two are pure
# rewordings); below it, pairs are mostly sibling topics sharing a prompt template.
EXCLUDE_AT = 0.9


def nearest(test_qs: list[str], other_qs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """For each test question: (index of most similar other question, its cosine)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(test_qs + other_qs)
    sims = (vec.transform(test_qs) @ vec.transform(other_qs).T).toarray()
    idx = sims.argmax(axis=1)
    return idx, sims[np.arange(len(test_qs)), idx]


def main(splits_path: str = "data/splits.json", out: str = "data/leak_check.json") -> dict:
    splits = json.loads(Path(splits_path).read_text())
    q_of = {r.answer_id: normalize_question(r.question) for r in load_records()}
    qs = {s: sorted({q_of[a] for a in splits[s]}) for s in ("train", "val", "test")}
    other = qs["train"] + qs["val"]
    idx, best = nearest(qs["test"], other)
    order = np.argsort(-best)
    leaked = {other[idx[i]] for i in np.flatnonzero(best >= EXCLUDE_AT)}
    exclude = sorted(a for s in ("train", "val") for a in splits[s] if q_of[a] in leaked)
    res = {
        "test_questions": len(qs["test"]),
        "train_val_questions": len(other),
        "above": {str(c): int((best >= c).sum()) for c in CUTOFFS},
        "exclude_at": EXCLUDE_AT,
        "exclude": exclude,  # train/val answers dropped before training
        "top_pairs": [
            {"sim": round(float(best[i]), 3), "test": qs["test"][i], "train_val": other[idx[i]]}
            for i in order[:25]
        ],
    }
    Path(out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k not in ("top_pairs", "exclude")}, indent=1))
    print(f"excluding {len(r['exclude'])} train/val answers")
    for p in r["top_pairs"]:
        print(f"{p['sim']:.3f} | {p['test'][:110]}\n      | {p['train_val'][:110]}")
