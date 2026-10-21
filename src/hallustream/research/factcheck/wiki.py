"""Wikipedia passages for one person, from FActScore's enwiki-20230401 SQLite DB.

The DB has one row per page: (title, text), where text is the page split into
~256-token passages joined by SEPARATOR. A judge reads every passage of a short
page, or the best-matching ones (BM25 against the biography) for a long page.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

DB_REPO = "danielli2003/enwiki-20230401.db"  # HF mirror of FActScore's Google Drive file
DB_FILE = "enwiki-20230401.db"
SEPARATOR = "####SPECIAL####SEPARATOR####"
MAX_PASSAGES = 24

_TOKEN = re.compile(r"\w+")


def clean(passage: str) -> str:
    return re.sub(r"</?s>", " ", passage).replace("  ", " ").strip()


class WikiDB:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)

    def passages(self, title: str) -> list[str]:
        row = self.conn.execute("SELECT text FROM documents WHERE title = ?", (title,)).fetchone()
        if row is None:
            raise KeyError(f"no Wikipedia page titled {title!r}")
        return [p for p in (clean(x) for x in row[0].split(SEPARATOR)) if p]


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def bm25_rank(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Indices of docs, best match first (Okapi BM25)."""
    toks = [tokens(d) for d in docs]
    avg = sum(map(len, toks)) / max(len(toks), 1)
    df = Counter(t for d in toks for t in set(d))
    n = len(docs)
    q = set(tokens(query))

    def score(d: list[str]) -> float:
        tf = Counter(d)
        s = 0.0
        for t in q:
            if tf[t]:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        return s

    scores = [score(d) for d in toks]
    return sorted(range(n), key=lambda i: -scores[i])


def evidence(passages: list[str], biography: str, limit: int = MAX_PASSAGES) -> list[str]:
    """All passages of a short page; the best `limit` (kept in page order) of a long one."""
    if len(passages) <= limit:
        return passages
    keep = sorted(bm25_rank(biography, passages)[:limit])
    return [passages[i] for i in keep]
