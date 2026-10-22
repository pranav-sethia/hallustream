"""Rewrite flagged sentences of a draft biography, in order, with Wikipedia evidence.

Every trigger is applied to the same draft: the draft's sentences are scored once, each
trigger picks which sentences to fix, and the fix rewrites those sentences one after another.
Each rewrite sees the text before it (with earlier rewrites applied), the question and
evidence passages, and generates one sentence; the rest of the draft is kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from hallustream.fix.spans import sentences

REWRITE_MAX_TOKENS = 96


@dataclass
class Draft:
    """A biography as alternating pieces: text between sentences, and sentences."""

    name: str
    gaps: list[str]  # len(sents) + 1 pieces of text around the sentences
    sents: list[str]
    fixed: dict[int, str] = field(default_factory=dict)  # sentence index -> new text ("" = dropped)

    @classmethod
    def of(cls, name: str, text: str) -> Draft:
        spans = sentences(text)
        gaps, sents, prev = [], [], 0
        for lo, hi in spans:
            gaps.append(text[prev:lo])
            sents.append(text[lo:hi])
            prev = hi
        gaps.append(text[prev:])
        return cls(name, gaps, sents)

    def text(self, upto: int | None = None) -> str:
        """The current text, or only what comes before sentence `upto`."""
        n = len(self.sents) if upto is None else upto
        parts = []
        for i in range(n):
            parts += [self.gaps[i], self.fixed.get(i, self.sents[i])]
        if upto is None:
            parts.append(self.gaps[-1])
        else:
            parts.append(self.gaps[upto])
        out = "".join(parts)
        return out.replace("\n\n\n", "\n\n") if upto is None else out


def first_sentence(generated: str) -> str:
    """The first sentence of generated text (a list item or line counts as one)."""
    line = generated.split("\n", 1)[0]
    spans = sentences(line)
    return line[: spans[0][1]] if spans else line.strip()


def evidence_message(question: str, passages: list[str]) -> str:
    listed = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return f"{question}\n\nHere are passages from Wikipedia that may help:\n\n{listed}"


def length_penalized(precision: float, n_claims: int, gamma: int = 10) -> float:
    """FActScore with the length penalty (Min et al. 2023): precision * exp(1 - gamma / n)."""
    if n_claims == 0:
        return 0.0
    return precision * (math.exp(1 - gamma / n_claims) if n_claims < gamma else 1.0)
