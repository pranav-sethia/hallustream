"""FActScore people lists and human labels (Min et al. 2023).

data.zip holds:
  labeled/{InstructGPT,ChatGPT,PerplexityAI}.jsonl  183 people each, with human-labeled atomic facts
  unlabeled/prompt_entities.txt                    500 people with no labels (our evaluation set)
The two lists do not overlap.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

DATA_ZIP_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=155exEdKs7R21gZF4G-x54-XN3qswBcPo&export=download&confirm=t"
)
LABELED_SYSTEMS = ("InstructGPT", "ChatGPT", "PerplexityAI")
BIO_PROMPT = "Tell me a bio of {name}."


@dataclass
class HumanLabeledBio:
    system: str
    name: str
    category: list[str]  # e.g. ["very rare", "North America"]
    output: str
    facts: list[tuple[str, str]]  # (atomic fact, "S" | "NS" | "IR")

    @property
    def claims(self) -> list[tuple[str, str]]:
        """Facts labeled S or NS (the labeling check leaves out irrelevant ones)."""
        return [f for f in self.facts if f[1] in ("S", "NS")]

    @property
    def factscore(self) -> float | None:
        """Supported facts over all facts; irrelevant ones count as unsupported, as in the paper."""
        if not self.facts:
            return None
        return sum(label == "S" for _, label in self.facts) / len(self.facts)


def fetch(dest: Path) -> Path:
    """Download and unpack data.zip into dest (skipped if already there)."""
    import urllib.request

    root = dest / "data"
    if not (root / "unlabeled" / "prompt_entities.txt").exists():
        dest.mkdir(parents=True, exist_ok=True)
        zpath = dest / "data.zip"
        urllib.request.urlretrieve(DATA_ZIP_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
    return root


def split_labeled(names: list[str], n_tune: int = 40) -> tuple[set[str], set[str]]:
    """Fixed split of the labeled people: a small set to pick judge settings, the rest to report."""
    import random

    order = sorted(set(names))
    random.Random(0).shuffle(order)
    return set(order[:n_tune]), set(order[n_tune:])


def eval_people(root: Path) -> list[str]:
    return [
        x.strip()
        for x in (root / "unlabeled" / "prompt_entities.txt").read_text().splitlines()
        if x.strip()
    ]


def human_labeled(root: Path, system: str) -> list[HumanLabeledBio]:
    out = []
    for line in (root / "labeled" / f"{system}.jsonl").read_text().splitlines():
        r = json.loads(line)
        facts = [
            (f["text"], f["label"])
            for sent in (r["annotations"] or [])
            for f in (sent.get("human-atomic-facts") or [])
            if f["label"] in ("S", "NS", "IR")
        ]
        out.append(HumanLabeledBio(system, r["topic"], r["cat"], r["output"], facts))
    return out
