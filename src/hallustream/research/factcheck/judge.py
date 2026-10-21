"""LLM judge for biographies: builds batch requests, parses answers, places entities in the text.

Two prompts (fixed in prompts/, see EVAL_PROTOCOL.md):
- bio_judge: label every entity and every atomic claim of a biography against Wikipedia passages.
- claim_judge: label a given list of claims (used to check the judge against human labels).
Requests are written as JSONL lines {"key", "request"} for the Gemini Batch API.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

PROMPTS = Path(__file__).parent / "prompts"
ENTITY_LABELS = ("Supported", "Not Supported", "Insufficient")
CLAIM_LABELS = ("Supported", "Not Supported")

# Judge settings compared on the tuning people; the chosen one is recorded in EVAL_PROTOCOL.md.
SETTINGS = {
    "t1_low": {"temperature": 1.0, "thinking": "low"},
    "t1_minimal": {"temperature": 1.0, "thinking": "minimal"},
    "t0_low": {"temperature": 0.0, "thinking": "low"},
    "t0_minimal": {"temperature": 0.0, "thinking": "minimal"},
}
MAX_OUTPUT_TOKENS = 16384


def _labeled(labels: tuple[str, ...], text_field: str, text_type: str = "STRING") -> dict:
    return {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                text_field: {"type": text_type},
                "note": {"type": "STRING"},
                "label": {"type": "STRING", "enum": list(labels)},
            },
            "required": [text_field, "note", "label"],
            "propertyOrdering": [text_field, "note", "label"],
        },
    }


BIO_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "entities": _labeled(ENTITY_LABELS, "span"),
        "claims": _labeled(CLAIM_LABELS, "text"),
    },
    "required": ["entities", "claims"],
    "propertyOrdering": ["entities", "claims"],
}
AUDIT_SCHEMA = {
    "type": "OBJECT",
    "properties": {"labels": _labeled(ENTITY_LABELS, "id", "INTEGER")},
    "required": ["labels"],
}
CLAIM_SCHEMA = {
    "type": "OBJECT",
    "properties": {"labels": _labeled(CLAIM_LABELS, "id", "INTEGER")},
    "required": ["labels"],
}


def _prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text()


def _passages(passages: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))


def _request(key: str, prompt: str, schema: dict, setting: str) -> dict:
    s = SETTINGS[setting]
    req = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {  # REST field names (camelCase)
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "temperature": s["temperature"],
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "thinkingConfig": {"thinkingLevel": s["thinking"]},
        },
    }
    return {"key": key, "request": req}


def bio_request(key, name, passages, biography, setting: str) -> dict:
    prompt = _prompt("bio_judge").format(
        name=name, passages=_passages(passages), biography=biography
    )
    return _request(key, prompt, BIO_SCHEMA, setting)


def claim_request(key, name, passages, claims, setting: str) -> dict:
    listed = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    prompt = _prompt("claim_judge").format(name=name, passages=_passages(passages), claims=listed)
    return _request(key, prompt, CLAIM_SCHEMA, setting)


def audit_request(key, name, passages, biography, items, setting: str) -> dict:
    """Re-judge chosen entities; items are (span, sentence) pairs."""
    listed = "\n".join(f'{i + 1}. "{span}" in: {sent}' for i, (span, sent) in enumerate(items))
    prompt = _prompt("entity_audit").format(
        name=name, passages=_passages(passages), biography=biography, entities=listed
    )
    return _request(key, prompt, AUDIT_SCHEMA, setting)


def parse_ids(text: str, n: int, labels: tuple[str, ...] = CLAIM_LABELS) -> list[str]:
    """One label per numbered item, joined on its id; ValueError if any is missing or repeated."""
    out: dict[int, str] = {}
    for x in _json(text).get("labels") or []:
        if not isinstance(x, dict) or x.get("label") not in labels:
            raise ValueError(f"bad label item: {x!r}")
        i = x.get("id")
        if not isinstance(i, int) or not 1 <= i <= n or i in out:
            raise ValueError(f"bad or repeated id: {i!r}")
        out[i] = x["label"]
    if len(out) != n:
        raise ValueError(f"{n - len(out)} of {n} items unlabeled")
    return [out[i] for i in range(1, n + 1)]


def sentence_of(text: str, index: int) -> str:
    """The sentence (or line) of text containing character index."""
    starts = [m.end() for m in re.finditer(r"(?<=[.!?])\s+|\n+", text[:index])]
    lo = starts[-1] if starts else 0
    m = re.search(r"(?<=[.!?])\s|\n", text[index:])
    hi = index + m.start() if m else len(text)
    return text[lo:hi].strip()


def response_text(row: dict) -> tuple[str | None, str | None]:
    """(answer text, finish reason) from one batch results line; text is None on errors.

    Thought parts are skipped and the remaining parts joined.
    """
    try:
        cand = row["response"]["candidates"][0]
    except (KeyError, IndexError, TypeError):
        return None, "ERROR"
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    return (text or None), cand.get("finishReason")


def _json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    d = json.loads(text)
    if not isinstance(d, dict):
        raise ValueError("judge answer is not a JSON object")
    return d


def _items(d: dict, field: str, text_field: str, labels: tuple[str, ...]) -> list[tuple]:
    items = d.get(field) or []
    if not isinstance(items, list):
        raise ValueError(f"{field} is not a list")
    return [
        (x[text_field], x["label"])
        for x in items
        if isinstance(x, dict)
        and isinstance(x.get(text_field), str)
        and x[text_field].strip()
        and x.get("label") in labels
    ]


def parse_bio(text: str) -> dict:
    """{"entities": [(span, label)], "claims": [(text, label)]}; malformed items are dropped.

    Raises ValueError (or json.JSONDecodeError) if the answer is not usable at all.
    """
    d = _json(text)
    return {
        "entities": _items(d, "entities", "span", ENTITY_LABELS),
        "claims": _items(d, "claims", "text", CLAIM_LABELS),
    }


def parse_claims(text: str, n: int) -> list[str]:
    """One label per claim, joined on the claim number.

    Raises ValueError unless every claim 1..n gets exactly one valid label, so a
    skipped or merged claim can never shift labels onto the wrong claim.
    """
    return parse_ids(text, n)


_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "‐": "-"})
_MARKDOWN = set("*`#")


def _normalize(text: str, keep_map: bool) -> tuple[str, list[int]]:
    """NFKC, folded quotes/dashes, no markdown symbols, collapsed whitespace.

    Also returns, for each output character, the index of the input character it came from.
    """
    out, where = [], []
    for i, ch in enumerate(text):
        if ch in _MARKDOWN:
            continue
        for c in unicodedata.normalize("NFKC", ch).translate(_FOLD):
            if c.isspace():
                if not out or out[-1] == " ":
                    continue
                c = " "
            out.append(c)
            where.append(i)
    return "".join(out), (where if keep_map else [])


def locate(biography: str, entities: list[tuple[str, str]]) -> tuple[list[dict], int]:
    """Character position of each entity, searching forward from the previous one.

    The judge lists entities in reading order, so the k-th "Paris" is matched to
    the k-th occurrence after the previous entity. Matching ignores markdown, quote
    and dash styles, needs word boundaries ("Paris" never matches inside "Parisian")
    and never reuses text already given to another entity. Returns (annotations,
    n_missed); annotations use the {"index", "span", "label"} format of the training
    data, with the span copied from the biography itself.
    """
    norm, where = _normalize(biography, keep_map=True)
    used: list[tuple[int, int]] = []
    out, missed, cursor = [], 0, 0
    for span, label in entities:
        target = _normalize(span, keep_map=False)[0].strip()
        if not target:
            missed += 1
            continue
        pat = re.compile(r"(?<!\w)" + re.escape(target) + r"(?!\w)")
        hit = None
        for start in (cursor, 0):
            for m in pat.finditer(norm, start):
                lo, hi = where[m.start()], where[m.end() - 1] + 1
                if all(hi <= a or lo >= b for a, b in used):
                    hit = (lo, hi, m.end())
                    break
            if hit:
                break
        if hit is None:
            missed += 1
            continue
        lo, hi, cursor = hit
        used.append((lo, hi))
        out.append({"index": lo, "span": biography[lo:hi], "label": label})
    return out, missed


def for_alignment(annotations: list[dict], insufficient_as_positive: bool) -> list[dict]:
    """Map judge labels onto align.LABEL_MAP names (primary analysis drops Insufficient)."""
    name = {"Supported": "Supported", "Not Supported": "Not Supported"}
    name["Insufficient"] = "Insufficient Information" if insufficient_as_positive else "N/A"
    return [a | {"label": name[a["label"]]} for a in annotations]
