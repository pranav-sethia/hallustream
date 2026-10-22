"""Sentences and entity spans in a biography, as character ranges."""

from __future__ import annotations

import random
import re
from functools import lru_cache

# Entity types the judge labels (EVAL_PROTOCOL.md): people, organizations, places, dates,
# quantities, titles of works, awards; plus the descriptive roles it often includes.
GLINER_LABELS = [
    "person", "organization", "location", "date", "number", "nationality",
    "occupation", "title of work", "award", "event",
]  # fmt: skip
GLINER_MODEL = "urchade/gliner_medium-v2.1"  # Apache-2.0
_SKIP_LINE = re.compile(r"^\s*(#+\s|[\s#*\-_>`|:]*$)")  # headers and markdown-only lines


def split_people(names: list[str], n_dev: int = 100, seed: int = 0) -> tuple[list, list]:
    """FActScore's people split once into dev and test (EVAL_PROTOCOL.md, detect-and-fix)."""
    order = sorted(names)
    random.Random(seed).shuffle(order)
    return order[:n_dev], order[n_dev:]


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


@lru_cache(maxsize=1)
def _gliner(device: str = "cpu"):
    from gliner import GLiNER

    return GLiNER.from_pretrained(GLINER_MODEL).to(device)


def entities(text: str, sents: list[tuple[int, int]], threshold: float = 0.4) -> list[tuple]:
    """(start, end) character spans of entities, found sentence by sentence with GLiNER."""
    model = _gliner()
    out = []
    for lo, hi in sents:
        for e in model.predict_entities(text[lo:hi], GLINER_LABELS, threshold=threshold):
            out.append((lo + e["start"], lo + e["end"]))
    return out


def to_tokens(offsets: list[tuple[int, int]], lo: int, hi: int) -> tuple[int, int] | None:
    """First and last token index overlapping characters [lo, hi), from tokenizer offsets."""
    toks = [i for i, (s, e) in enumerate(offsets) if e > s and s < hi and e > lo]
    return (toks[0], toks[-1]) if toks else None
