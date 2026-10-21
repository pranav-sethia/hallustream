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
    claims: list[tuple[str, str]]  # (atomic fact, "S" | "NS"); irrelevant facts dropped

    @property
    def factscore(self) -> float | None:
        if not self.claims:
            return None
        return sum(label == "S" for _, label in self.claims) / len(self.claims)


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
        claims = [
            (f["text"], f["label"])
            for sent in (r["annotations"] or [])
            for f in (sent.get("human-atomic-facts") or [])
            if f["label"] in ("S", "NS")
        ]
        out.append(HumanLabeledBio(system, r["topic"], r["cat"], r["output"], claims))
    return out
