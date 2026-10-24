"""Read SHARS's generation log: what happened to every sentence it tried.

SHARS prints one block per attempted sentence (generator.py): the sentence, each proposition
with its uncertainty and verdict, then keep / rewrite / discard. This turns a log into counts
and a list of rewrites, and checks each rewrite against the propositions it was built from.
"""

from __future__ import annotations

import re
from collections import Counter

PROP = re.compile(r"^Prop-\d+ \(Uncertainty:\s*(\S+), (Hallued|Factual)\) :\t(.*)\t\((.*)\)$")
ENDS = {
    "complete due to EOS": "eos",
    "max number of resample attempts": "max_resamples",
    "max number of repetition attempts": "max_repetitions",
    "has no knowledge": "non_answer",
    "unexpected error": "error",
    "max new tokens": "max_tokens",
}


def parse(log: str) -> dict:
    """Counts of sentence outcomes, the rewrites, and how generation ended."""
    lines = log.splitlines()
    counts: Counter = Counter()
    rewrites, props, end = [], [], None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Attempting "):
            counts["attempts"] += 1
            props = []
        elif m := PROP.match(lines[i].strip("\n")):
            props.append({"uncertainty": float(m[1]), "flagged": m[2] == "Hallued",
                          "claim": m[3], "entity": m[4]})  # fmt: skip
            counts["props"] += 1
            counts["props_flagged"] += m[2] == "Hallued"
        elif line == "All propositions are factual, keeping the sentence.":
            counts["kept"] += 1
        elif line == "All propositions are hallucinated.":
            counts["all_flagged"] += 1
        elif line == "No factual propositions are found in the sentence.":
            counts["no_props"] += 1
        elif line.startswith("Resample as repeated sentence"):
            counts["repeats"] += 1
        elif line == "Rewrite the original sentence:":
            original = lines[i + 1].strip() if i + 1 < len(lines) else ""
            nxt = lines[i + 2].strip() if i + 2 < len(lines) else ""
            if nxt == "to the new sentence:":
                new = lines[i + 3].strip() if i + 3 < len(lines) else ""
                dup = i + 4 < len(lines) and lines[i + 4].startswith("Discard the rewritten")
                rewrites.append(_check(original, new, props) | {"duplicate": dup})
                counts["rewrite_duplicate" if dup else "rewritten"] += 1
                i += 3
            else:
                counts["rewrite_failed"] += 1
                i += 1
        elif line.startswith("[END]"):
            end = next((v for k, v in ENDS.items() if k in line), "other")
        i += 1
    return {"counts": dict(counts), "rewrites": rewrites, "end": end}


def _check(original: str, new: str, props: list[dict]) -> dict:
    """Does the rewrite drop every flagged entity and keep the trusted ones?"""
    low = new.lower()
    flagged = [p["entity"] for p in props if p["flagged"]]
    trusted = [p["entity"] for p in props if not p["flagged"]]
    return {
        "original": original,
        "new": new,
        "trusted_claims": [p["claim"] for p in props if not p["flagged"]],
        "flagged_kept": [e for e in flagged if e.lower() in low],
        "trusted_missing": [e for e in trusted if e.lower() not in low],
    }
