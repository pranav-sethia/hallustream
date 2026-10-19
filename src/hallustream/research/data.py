"""Load the annotated datasets and build question-grouped train/val/test splits.

The same questions were answered by several source models, so we split by
question: any train/val answer whose (normalized) question appears in the test
set is removed. That way the probe is never tested on a question it trained on.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

DATASETS = {
    # name -> (HF repo, pinned revision)
    "longfact": (
        "obalcells/longfact-annotations",
        "949fd2f71b3efd403350e38539340d9d9ceefc9d",
    ),
    "longfact_aug": (
        "obalcells/longfact-augmented-annotations",
        "63174c7ea9111adb6f4c4ea61895af3e3740d997",
    ),
}
SOURCES = [
    "Qwen2.5-7B-Instruct",
    "Mistral-Small-24B-Instruct-2501",
    "gemma-2-9b-it",
    "Meta-Llama-3.1-8B-Instruct",
    "Llama-3.3-70B-Instruct",
]
SPLITS = ["train", "validation", "test"]


@dataclass
class Record:
    answer_id: str
    dataset: str  # "longfact" | "longfact_aug"
    source: str  # model that wrote the answer
    orig_split: str  # official split
    subset: str
    messages: list[dict]  # conversation before the answer (user turn, maybe system)
    answer: str
    annotations: list[dict] = field(default_factory=list)

    @property
    def question(self) -> str:
        return " ".join(m["content"] for m in self.messages if m["role"] == "user")


def normalize_question(text: str) -> str:
    """Lowercase and collapse whitespace so trivially different copies match."""
    return re.sub(r"\s+", " ", text).strip().lower()


def question_key(text: str) -> str:
    return hashlib.sha1(normalize_question(text).encode()).hexdigest()[:16]


def load_records(sources: list[str] = SOURCES) -> list[Record]:
    from datasets import load_dataset

    records = []
    for ds_name, (repo, rev) in DATASETS.items():
        for source in sources:
            for split in SPLITS:
                rows = load_dataset(repo, source, split=split, revision=rev)
                for i, row in enumerate(rows):
                    conv = row["conversation"]
                    records.append(
                        Record(
                            answer_id=f"{ds_name}/{source}/{split}/{i}",
                            dataset=ds_name,
                            source=source,
                            orig_split=split,
                            subset=row["subset"],
                            messages=conv[:-1],
                            answer=conv[-1]["content"],
                            annotations=row["annotations"],
                        )
                    )
    return records


@dataclass
class SplitResult:
    train: list[str]
    val: list[str]
    test: list[str]
    stats: dict


def make_splits(records: list[Record], train_sources: list[str]) -> SplitResult:
    """Test = every official test answer. Train/val = official splits of
    `train_sources`, minus any answer whose question appears in test.
    A question that lands in both train and val is kept in val only."""
    test = [r for r in records if r.orig_split == "test"]
    test_keys = {question_key(r.question) for r in test}

    def candidates(split: str) -> list[Record]:
        return [r for r in records if r.orig_split == split and r.source in train_sources]

    train_c, val_c = candidates("train"), candidates("validation")
    val = [r for r in val_c if question_key(r.question) not in test_keys]
    val_keys = {question_key(r.question) for r in val}
    train = [r for r in train_c if question_key(r.question) not in test_keys | val_keys]

    stats = {
        "train_candidates": len(train_c),
        "train_kept": len(train),
        "train_removed_frac": round(1 - len(train) / max(len(train_c), 1), 4),
        "val_candidates": len(val_c),
        "val_kept": len(val),
        "test": len(test),
        "by_source": {
            s: {
                "train": sum(r.source == s for r in train),
                "val": sum(r.source == s for r in val),
                "test": sum(r.source == s for r in test),
            }
            for s in SOURCES
        },
    }
    return SplitResult(
        train=[r.answer_id for r in train],
        val=[r.answer_id for r in val],
        test=[r.answer_id for r in test],
        stats=stats,
    )


def save_splits(result: SplitResult, path: Path, train_sources: list[str]) -> None:
    payload = {
        "datasets": {k: {"repo": v[0], "revision": v[1]} for k, v in DATASETS.items()},
        "train_sources": train_sources,
        **asdict(result),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/splits.json")
    recs = load_records()
    result = make_splits(recs, train_sources=SOURCES)
    save_splits(result, out, train_sources=SOURCES)
    print(json.dumps({k: v for k, v in result.stats.items() if k != "by_source"}, indent=1))
