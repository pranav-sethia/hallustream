"""Teacher-forced feature extraction for one answer.

Runs the model once over prompt + answer and returns:
- hidden states at the chosen layers for every entity token plus a random sample
  of ordinary answer tokens (each weighted to stand for the ones not sampled),
- per-token baseline signals (surprise = negative log-likelihood, entropy),
- span-level baseline scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hallustream.research.align import Aligned, align_exact
from hallustream.text import LOGIT_CHUNK, token_baselines  # noqa: F401  (re-exported)


@dataclass
class AnswerFeatures:
    rows: dict  # column -> list, one entry per saved token
    features: dict[int, np.ndarray]  # layer -> (n_saved, hidden) float16
    spans: list[dict]  # one per labeled entity, with baseline scores


def choose_rows(aligned: Aligned, n_neg: int, rng: np.random.Generator) -> tuple[list, list]:
    """All entity tokens, plus up to n_neg sampled ordinary answer tokens.

    Returns (positions, weights). A sampled ordinary token's weight is
    (#ordinary tokens / #sampled), so sums over samples estimate full sums.
    """
    entity = sorted({t for s in aligned.spans for t in range(s.start, s.end + 1)})
    entity_set = set(entity)
    ordinary = [
        t for t in range(aligned.answer_start, len(aligned.input_ids)) if t not in entity_set
    ]
    k = min(n_neg, len(ordinary))
    sampled = sorted(rng.choice(ordinary, size=k, replace=False).tolist()) if k else []
    w = len(ordinary) / k if k else 0.0
    return entity + sampled, [1.0] * len(entity) + [w] * len(sampled)


@torch.no_grad()
def extract_answer(
    model,
    tokenizer,
    record,
    layers: list[int],
    n_neg: int,
    rng: np.random.Generator,
    split: str,
    **chat_kwargs,
) -> AnswerFeatures:
    a = align_exact(tokenizer, record.messages, record.answer, record.annotations, **chat_kwargs)
    ids = torch.tensor(a.input_ids, device=model.device)
    out = model(ids[None], output_hidden_states=True)
    hs = out.hidden_states  # index k = output of decoder layer k-1
    nll, ent = token_baselines(out.logits[0], ids)

    positions, weights = choose_rows(a, n_neg, rng)
    pos_t = torch.tensor(positions, device=model.device, dtype=torch.long)
    feats = {layer: hs[layer + 1][0, pos_t].to(torch.float16).cpu().numpy() for layer in layers}
    if any(not np.isfinite(f).all() for f in feats.values()):
        raise FloatingPointError(f"non-finite hidden states in {record.answer_id}")

    token_span = {t: i for i, s in enumerate(a.spans) for t in range(s.start, s.end + 1)}
    rows = {
        "answer_id": [record.answer_id] * len(positions),
        "split": [split] * len(positions),
        "source": [record.source] * len(positions),
        "dataset": [record.dataset] * len(positions),
        "token_pos": positions,
        "label": [a.labels[t] for t in positions],
        "weight": weights,
        "span_id": [token_span.get(t, -1) for t in positions],
        "nll": nll[positions].tolist(),
        "entropy": ent[positions].tolist(),
    }
    spans = []
    for i, s in enumerate(a.spans):
        sl = slice(s.start, s.end + 1)
        spans.append(
            {
                "answer_id": record.answer_id,
                "split": split,
                "source": record.source,
                "dataset": record.dataset,
                "subset": record.subset,
                "span_id": i,
                "start": s.start,  # token range, inclusive (spans may overlap)
                "end": s.end,
                "label": s.label,
                "n_tokens": s.end - s.start + 1,
                "nll_max": float(np.nanmax(nll[sl])),
                "nll_mean": float(np.nanmean(nll[sl])),
                "entropy_max": float(np.nanmax(ent[sl])),
                "entropy_mean": float(np.nanmean(ent[sl])),
            }
        )
    return AnswerFeatures(rows, feats, spans)
