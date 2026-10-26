"""Find the entities to score: names, places, dates, numbers, titles. Uses GLiNER, the span
finder chosen in the detect-and-fix pilot (EVAL_PROTOCOL.md); install with
`pip install "hallustream[entities]"` (Python 3.11+), or pass your own spans."""

from __future__ import annotations

from functools import lru_cache

# Entity types the judge labels (EVAL_PROTOCOL.md): people, organizations, places, dates,
# quantities, titles of works, awards; plus the descriptive roles it often includes.
GLINER_LABELS = [
    "person", "organization", "location", "date", "number", "nationality",
    "occupation", "title of work", "award", "event",
]  # fmt: skip
GLINER_MODEL = "urchade/gliner_medium-v2.1"  # Apache-2.0


def available() -> bool:
    try:
        import gliner  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def _gliner(device: str = "cpu"):
    from gliner import GLiNER  # pyright: ignore[reportMissingImports]  (Python 3.11+ only)

    return GLiNER.from_pretrained(GLINER_MODEL).to(device)


def find(text: str, sents: list[tuple[int, int]], threshold: float = 0.4) -> list[tuple]:
    """(start, end) character spans of entities, found sentence by sentence with GLiNER."""
    if not available():
        raise ImportError(
            "finding entities needs GLiNER: pip install 'hallustream[entities]' (Python 3.11+),"
            " or pass spans=[(start, end), ...] yourself"
        )
    model = _gliner()
    out = []
    for lo, hi in sents:
        for e in model.predict_entities(text[lo:hi], GLINER_LABELS, threshold=threshold):
            out.append((lo + e["start"], lo + e["end"]))
    return out
