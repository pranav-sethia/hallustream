"""The probe: per-token probabilities from one hidden layer, and entity scores.

v1 is the frozen probe (mean over seeds of each seed's max over an entity's tokens); v3 adds a
linear term on the entity's mean hidden state, fitted on the model's own answers. Scoring is
by teacher forcing, which is how the probe's training data was produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

EPS = 1e-6


@dataclass
class Forward:
    """One teacher-forced answer. Token arrays cover the whole text, prompt included."""

    ids: list[int]
    offsets: list[tuple[int, int]]  # character offsets in the answer (-1, -1 for prompt tokens)
    answer_start: int
    probs: np.ndarray  # (T, n_seeds) probe probability per token and seed
    hidden: np.ndarray  # (T, d) hidden state at the probe's layer, float16
    nll: np.ndarray  # (T,) surprise of each token (NaN at position 0)
    entropy: np.ndarray  # (T,)


class Probe:
    """v1 (frozen probe seeds) and v3 (v1 plus a hidden-state term) from their saved weights.
    The probe reads the output of decoder layer `layer` (hidden_states[layer + 1])."""

    def __init__(self, probe_files: list[Path], v3_json: Path, device: str = "cuda"):
        from safetensors.torch import load_file

        ps = [load_file(str(f), device=device) for f in probe_files]
        self.weight = torch.stack([p["weight"] for p in ps], dim=1).float()  # (d, n_seeds)
        self.bias = torch.stack([p["bias"].reshape(()) for p in ps]).float()  # (n_seeds,)
        det = json.loads(Path(v3_json).read_text())["detector"]
        assert det["candidate"] == "B" and det["offset"]
        self.layer = int(det["layer"])
        self.v3_w = np.asarray(det["w"], dtype=np.float32)
        self.v3_b = float(det["b"])
        self.h_mean = np.asarray(det["h_mean"], dtype=np.float32)
        self.h_std = np.asarray(det["h_std"], dtype=np.float32)

    def token_probs(self, h: torch.Tensor) -> np.ndarray:
        return torch.sigmoid(h.float() @ self.weight + self.bias).cpu().numpy()

    def span_scores(self, fw: Forward, spans: list[tuple[int, int]]) -> tuple[np.ndarray, ...]:
        """(v1 logit, v3 logit) per token span (first, last inclusive), as in training:
        v1 = mean over seeds of each seed's max over the span."""
        if not spans:
            return np.zeros(0), np.zeros(0)
        p = np.stack([fw.probs[a : b + 1].max(0) for a, b in spans]).mean(1)
        p = np.clip(p, EPS, 1 - EPS)
        v1 = np.log(p / (1 - p))
        h = np.stack([fw.hidden[a : b + 1].astype(np.float32).mean(0) for a, b in spans])
        v3 = v1 + self.v3_b + ((h - self.h_mean) / self.h_std) @ self.v3_w
        return v1, v3


@torch.no_grad()
def forward(lm, tok, probe: Probe, messages: list[dict], answer: str, **chat) -> Forward:
    from hallustream.text import build_text, token_baselines

    text, answer_char = build_text(tok, messages, answer, **chat)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = [
        (s - answer_char, e - answer_char) if s >= answer_char else (-1, -1)
        for s, e in enc["offset_mapping"]
    ]
    answer_start = next(i for i, (_, e) in enumerate(enc["offset_mapping"]) if e > answer_char)
    x = torch.tensor([ids], device=lm.device)
    out = lm(x, output_hidden_states=True)
    h = out.hidden_states[probe.layer + 1][0]
    nll, ent = token_baselines(out.logits[0], x[0])
    return Forward(ids, offsets, answer_start, probe.token_probs(h),
                   h.half().cpu().numpy(), nll, ent)  # fmt: skip


def char_to_tokens(fw: Forward, lo: int, hi: int) -> tuple[int, int] | None:
    """First and last answer token overlapping answer characters [lo, hi)."""
    toks = [i for i, (s, e) in enumerate(fw.offsets) if e > s >= 0 and s < hi and e > lo]
    return (toks[0], toks[-1]) if toks else None
