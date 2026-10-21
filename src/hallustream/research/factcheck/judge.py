"""LLM judge for biographies: builds batch requests, parses answers, places entities in the text.

Two prompts (fixed in prompts/, see EVAL_PROTOCOL.md):
- bio_judge: label every entity and every atomic claim of a biography against Wikipedia passages.
- claim_judge: label a given list of claims (used to check the judge against human labels).
Requests are written as JSONL lines {"key", "request"} for the Gemini Batch API.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROMPTS = Path(__file__).parent / "prompts"
ENTITY_LABELS = ("Supported", "Not Supported", "Insufficient")
CLAIM_LABELS = ("Supported", "Not Supported")


def _prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text()


def _passages(passages: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))


def _request(key: str, prompt: str, generation_config: dict | None) -> dict:
    req = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        # REST field names (camelCase); e.g. generation_config={"thinkingConfig": {...}}
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0}
        | (generation_config or {}),
    }
    return {"key": key, "request": req}


def bio_request(key, name, passages, biography, generation_config=None) -> dict:
    prompt = _prompt("bio_judge").format(
        name=name, passages=_passages(passages), biography=biography
    )
    return _request(key, prompt, generation_config)


def claim_request(key, name, passages, claims, generation_config=None) -> dict:
    listed = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    prompt = _prompt("claim_judge").format(name=name, passages=_passages(passages), claims=listed)
    return _request(key, prompt, generation_config)


def _json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


def parse_bio(text: str) -> dict:
    """{"entities": [(span, label)], "claims": [(text, label)]}; unknown labels are dropped."""
    d = _json(text)
    ents = [
        (e["span"], e["label"])
        for e in d.get("entities", [])
        if isinstance(e, dict) and e.get("span") and e.get("label") in ENTITY_LABELS
    ]
    claims = [
        (c["text"], c["label"])
        for c in d.get("claims", [])
        if isinstance(c, dict) and c.get("text") and c.get("label") in CLAIM_LABELS
    ]
    return {"entities": ents, "claims": claims}


def parse_claims(text: str, n: int) -> list[str | None]:
    """One label per claim; None where the judge's answer is missing or malformed."""
    labels = _json(text).get("labels", [])
    out = [lab if lab in CLAIM_LABELS else None for lab in labels[:n]]
    return out + [None] * (n - len(out))


def locate(biography: str, entities: list[tuple[str, str]]) -> tuple[list[dict], int]:
    """Character position of each entity, searching forward from the previous one.

    The judge lists entities in reading order, so the k-th "Paris" is matched to
    the k-th occurrence after the previous entity. Returns (annotations, n_missed);
    annotations use the {"index", "span", "label"} format of the training data.
    """
    out, missed, cursor = [], 0, 0
    for span, label in entities:
        i = biography.find(span, cursor)
        if i < 0:
            i = biography.find(span)  # listed out of order
        if i < 0:
            missed += 1
            continue
        out.append({"index": i, "span": span, "label": label})
        cursor = i + len(span)
    return out, missed


def for_alignment(annotations: list[dict], insufficient_as_positive: bool) -> list[dict]:
    """Map judge labels onto align.LABEL_MAP names (primary analysis drops Insufficient)."""
    name = {"Supported": "Supported", "Not Supported": "Not Supported"}
    name["Insufficient"] = "Insufficient Information" if insufficient_as_positive else "N/A"
    return [a | {"label": name[a["label"]]} for a in annotations]
