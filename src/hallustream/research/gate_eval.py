"""Reproduction gate, CPU part.

Aligns labels both ways ("paper" replica and our "exact" offsets) on the saved
per-token probabilities and compares AUROC with the paper's Table 4.

Run: uv run python -m hallustream.research.gate_eval data/gate/probs.npz
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

from hallustream.research.align import align_exact, align_paper, build_text
from hallustream.research.data import load_records
from hallustream.research.metrics import auroc, recall_at_fpr, span_scores

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PAPER = {"longfact": 0.8383, "longfact_aug": 0.8467}  # Obeso et al. Table 4, linear probe
TOLERANCE = 0.02


def main(npz_path: str) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    probs_by_id = np.load(npz_path)
    recs = [r for r in load_records(sources=["Qwen2.5-7B-Instruct"]) if r.orig_split == "test"]
    results = {}
    for ds in PAPER:
        out = {}
        for mode in ("paper", "exact"):
            all_scores, all_labels, stats = [], [], Counter()
            for r in (r for r in recs if r.dataset == ds):
                probs = probs_by_id[r.answer_id.replace("/", "__")]
                if mode == "exact":
                    a = align_exact(tok, r.messages, r.answer, r.annotations)
                    spans, st = a.spans, a.stats
                else:
                    text, _ = build_text(tok, r.messages, r.answer)
                    ids = tok(text, add_special_tokens=False).input_ids
                    spans, st = align_paper(tok, ids, r.answer, r.annotations)
                stats.update(st)
                if spans:
                    s, lab = span_scores(probs, spans)
                    all_scores.append(s)
                    all_labels.append(lab)
            scores, labels = np.concatenate(all_scores), np.concatenate(all_labels)
            auc = auroc(labels, scores)
            out[mode] = {
                "auroc": round(auc, 4),
                "recall_at_10pct_fpr": round(recall_at_fpr(labels, scores), 4),
                "n_spans": int(len(labels)),
                "positive_rate": round(float(labels.mean()), 4),
                "stats": dict(stats),
            }
        paper_auc = PAPER[ds]
        out["paper_reported"] = paper_auc
        out["paper_mode_diff"] = round(out["paper"]["auroc"] - paper_auc, 4)
        out["gate_pass"] = abs(out["paper_mode_diff"]) <= TOLERANCE
        results[ds] = out
    return results


if __name__ == "__main__":
    res = main(sys.argv[1] if len(sys.argv) > 1 else "data/gate/probs.npz")
    print(json.dumps(res, indent=2))
    with open("data/gate/gate_results.json", "w") as f:
        json.dump(res, f, indent=2)
