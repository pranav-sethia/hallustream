"""Text helpers: the prompt and answer as the model sees them, per-token confidence signals,
sentences, and character-to-token spans."""

from __future__ import annotations

import re

import numpy as np
import torch

END_OF_TURN = "<|im_end|>\n"
LOGIT_CHUNK = 512  # positions per log-softmax chunk, to bound memory
_SKIP_LINE = re.compile(r"^\s*(#+\s|[\s#*\-_>`|:]*$)")  # headers and markdown-only lines


def build_text(tokenizer, messages: list[dict], answer: str, **chat_kwargs) -> tuple[str, int]:
    """Prompt exactly as the model sees it when it starts answering, then the answer.

    Returns the full text and the character position where the answer begins.
    """
    prefix = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **chat_kwargs
    )
    return prefix + answer + END_OF_TURN, len(prefix)


def token_baselines(logits: torch.Tensor, input_ids: torch.Tensor) -> tuple[np.ndarray, ...]:
    """Surprise (NLL of the actual next token) and predictive entropy per position.

    Position t is scored with the logits at t-1, so position 0 gets NaN.
    """
    n = input_ids.shape[0]
    nll = np.full(n, np.nan, dtype=np.float32)
    ent = np.full(n, np.nan, dtype=np.float32)
    for a in range(0, n - 1, LOGIT_CHUNK):
        b = min(a + LOGIT_CHUNK, n - 1)
        logp = torch.log_softmax(logits[a:b].float(), dim=-1)
        nxt = input_ids[a + 1 : b + 1]
        nll[a + 1 : b + 1] = (-logp.gather(-1, nxt[:, None]).squeeze(-1)).cpu().numpy()
        ent[a + 1 : b + 1] = (-(logp.exp() * logp).sum(-1)).cpu().numpy()
    return nll, ent


def sentences(text: str) -> list[tuple[int, int]]:
    """Character ranges of sentences; newlines always end one; headers and markdown-only
    lines are skipped."""
    import pysbd

    seg = pysbd.Segmenter(language="en", clean=False, char_span=True)
    out = []
    for m in re.finditer(r"[^\n]+", text):
        line, base = m.group(), m.start()
        if _SKIP_LINE.match(line):
            continue
        for s in seg.segment(line):
            lo, hi = base + s.start, base + s.end  # pyright: ignore[reportAttributeAccessIssue]
            while hi > lo and text[hi - 1].isspace():
                hi -= 1
            if hi > lo:
                out.append((lo, hi))
    return out


def to_tokens(offsets: list[tuple[int, int]], lo: int, hi: int) -> tuple[int, int] | None:
    """First and last token index overlapping characters [lo, hi), from tokenizer offsets."""
    toks = [i for i, (s, e) in enumerate(offsets) if e > s and s < hi and e > lo]
    return (toks[0], toks[-1]) if toks else None
