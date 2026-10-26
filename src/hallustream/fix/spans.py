"""Sentences and entity spans in a biography, as character ranges."""

from __future__ import annotations

import random

from hallustream.entities import GLINER_LABELS, GLINER_MODEL
from hallustream.entities import find as entities
from hallustream.text import sentences, to_tokens

__all__ = ["GLINER_LABELS", "GLINER_MODEL", "entities", "sentences", "split_people", "to_tokens"]


def split_people(names: list[str], n_dev: int = 100, seed: int = 0) -> tuple[list, list]:
    """FActScore's people split once into dev and test (EVAL_PROTOCOL.md, detect-and-fix)."""
    order = sorted(names)
    random.Random(seed).shuffle(order)
    return order[:n_dev], order[n_dev:]
