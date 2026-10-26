"""Score a written answer: per-token probe and confidence signals, then spans and sentences.

One teacher-forced forward pass gives everything the triggers need. Scoring by teacher
forcing matches how the probe's training data was produced.
"""

from __future__ import annotations

import numpy as np

from hallustream.probe import EPS, Forward, char_to_tokens, forward
from hallustream.probe import Probe as Detector

LAYER = 21  # Qwen3-4B's probe layer (the library reads it from the weights)
__all__ = ["EPS", "LAYER", "Detector", "Forward", "char_to_tokens", "forward", "sentence_signals"]


def sentence_signals(fw: Forward, det: Detector, sent: tuple[int, int], ents: list) -> dict:
    """Trigger signals for one sentence: v3/v1 max over its entity spans, and max over its
    tokens of the probe, the surprise (FLARE-style low probability) and the entropy."""
    tok_span = char_to_tokens(fw, *sent)
    if tok_span is None:
        return {}
    a, b = tok_span
    spans = [t for lo, hi in ents if sent[0] <= lo < sent[1] and (t := char_to_tokens(fw, lo, hi))]
    v1, v3 = det.span_scores(fw, spans)
    return {
        "v3_max": float(v3.max()) if len(v3) else -np.inf,
        "v1_max": float(v1.max()) if len(v1) else -np.inf,
        "probe_token_max": float(fw.probs[a : b + 1].mean(1).max()),
        "surprise_max": float(np.nanmax(fw.nll[a : b + 1])),
        "entropy_max": float(np.nanmax(fw.entropy[a : b + 1])),
        "n_entities": len(spans),
    }
