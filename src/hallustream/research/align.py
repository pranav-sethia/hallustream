"""Turn character-level entity labels into token-level labels.

Two modes:
- ``align_exact``: uses each annotation's character offset (``index``) and the
  tokenizer's offset mapping. Exact, and always labels the right mention. Used
  for all of our training and evaluation.
- ``align_paper``: a faithful replica of Obeso et al.'s string search (first
  decoded occurrence, searching forward from the previous match's start). Used
  only to reproduce their published numbers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from hallustream.text import END_OF_TURN, build_text  # noqa: F401  (re-exported)

LABEL_MAP: dict[str | None, int | None] = {
    "Supported": 0,
    "S": 0,
    "Not Supported": 1,
    "NS": 1,
    "Insufficient Information": 1,
    "N/A": None,
    None: None,
}
IGNORE = -100


@dataclass
class Span:
    start: int  # first token (inclusive)
    end: int  # last token (inclusive)
    label: int  # 1 = hallucinated, 0 = supported


@dataclass
class Aligned:
    input_ids: list[int]
    labels: list[int]  # per token: IGNORE (prompt), 0, or 1
    answer_start: int  # index of the first answer token
    spans: list[Span]
    stats: Counter = field(default_factory=Counter)


def _locate(answer: str, span: str, index: int | None) -> tuple[int | None, str]:
    """Character start of the span in the answer, and how it was found."""
    if index is not None and answer[index : index + len(span)] == span:
        return index, "exact"
    hits = [i for i in range(len(answer)) if answer.startswith(span, i)]
    if not hits:
        return None, "not_found"
    if index is None:
        return None, "no_index"
    return min(hits, key=lambda i: abs(i - index)), "recovered"


def align_exact(
    tokenizer,
    messages: list[dict],
    answer: str,
    annotations: list[dict],
    max_length: int = 4096,
    **chat_kwargs,
) -> Aligned:
    text, answer_char = build_text(tokenizer, messages, answer, **chat_kwargs)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    stats: Counter = Counter()

    answer_start = next(i for i, (_, e) in enumerate(offsets) if e > answer_char)
    if answer_start >= max_length:
        raise ValueError(f"prompt is {answer_start} tokens, longer than max_length={max_length}")
    spans: list[Span] = []
    for ann in annotations:
        label = LABEL_MAP[ann.get("label")]
        span_text = ann.get("span") or ""
        if label is None or not span_text:
            stats["ignored_label"] += 1
            continue
        start_char, how = _locate(answer, span_text, ann.get("index"))
        stats[how] += 1
        if start_char is None:
            continue
        lo, hi = answer_char + start_char, answer_char + start_char + len(span_text)
        toks = [i for i, (s, e) in enumerate(offsets) if e > s and s < hi and e > lo]
        if not toks or toks[-1] >= max_length:
            stats["dropped"] += 1
            continue
        spans.append(Span(toks[0], toks[-1], label))

    ids = ids[:max_length]
    labels = [IGNORE] * answer_start + [0] * (len(ids) - answer_start)
    for sp in sorted(spans, key=lambda s: s.label):  # positives last, so they win overlaps
        for t in range(sp.start, sp.end + 1):
            labels[t] = sp.label
    stats["kept"] = len(spans)
    return Aligned(ids, labels, answer_start, spans, stats)


# --- Faithful replica of the paper's span search (reproduction gate only) ---


def find_string_in_tokens(target: str, ids: list[int], tokenizer) -> tuple[int, int]:
    """Smallest token window [start, stop) whose decoded text contains `target`,
    taking the first occurrence (two binary searches over decoded prefixes)."""

    def dec(a: int, b: int) -> str:
        return tokenizer.decode(ids[a:b])

    if target not in dec(0, len(ids)):
        raise ValueError("not found")
    lo, hi = 0, len(ids)
    while lo < hi:  # smallest stop with target in decode(ids[:stop])
        mid = (lo + hi) // 2
        if target in dec(0, mid):
            hi = mid
        else:
            lo = mid + 1
    stop = lo
    lo, hi = 0, stop - 1
    while lo < hi:  # largest start with target in decode(ids[start:stop])
        mid = (lo + hi + 1) // 2
        if target in dec(mid, stop):
            lo = mid
        else:
            hi = mid - 1
    if target not in dec(lo, stop):
        raise ValueError("not found")
    return lo, stop


def align_paper(
    tokenizer, ids: list[int], answer: str, annotations: list[dict], max_length: int = 1536
) -> tuple[list[Span], Counter]:
    """Replicates obalcells/hallucination_probes probe/dataset.py span handling."""
    stats: Counter = Counter()
    ids = ids[:max_length]
    input_str = tokenizer.decode(ids)
    _, cur = find_string_in_tokens("<|im_start|>assistant", ids, tokenizer)

    kept = []
    for ann in annotations:
        span_text, idx = ann.get("span") or "", ann.get("index")
        if not isinstance(idx, int) or not span_text or span_text not in answer:
            stats["dropped_by_converter"] += 1
            continue
        kept.append((idx, span_text, LABEL_MAP[ann.get("label")]))

    spans: list[Span] = []
    for _, span_text, label in sorted(kept, key=lambda x: x[0]):
        if span_text not in input_str:
            stats["skipped"] += 1
            continue
        try:
            s, e = find_string_in_tokens(span_text, ids[cur:], tokenizer)
            s, e = s + cur, e + cur
        except ValueError:
            try:
                s, e = find_string_in_tokens(span_text, ids, tokenizer)
            except ValueError:
                stats["skipped"] += 1
                continue
        cur = s  # the paper resumes from the previous match's start
        if label is not None:
            spans.append(Span(s, e - 1, label))
    stats["kept"] = len(spans)
    return spans, stats
