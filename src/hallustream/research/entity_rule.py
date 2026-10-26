"""Effect of entities.is_specific on the library check's automatic-mode entities.

Re-runs GLiNER (CPU) on the same texts to get each entity's type, then compares all entities
with the specific ones: entities and flags per text, judge-false entities still found, AUROC.
The risks are unchanged by the rule, so no GPU is needed.

Run: uv run python -m hallustream.research.entity_rule results/library <texts dir>
(texts dir: topics/bios/qwen3/shard*.jsonl and fix/shars/naive/bios.jsonl from the volume)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from hallustream import entities
from hallustream.research.metrics import auroc
from hallustream.text import PRONOUNS, sentences

BALANCED = 0.737832


def kinds(names: list[str], texts: dict[str, str]) -> pd.DataFrame:
    model = entities._gliner("cpu")
    rows = []
    for n in names:
        for lo, hi in sentences(texts[n]):
            for e in model.predict_entities(texts[n][lo:hi], entities.GLINER_LABELS, threshold=0.4):
                rows.append({"name": n, "start": lo + e["start"], "end": lo + e["end"],
                             "kind": e["label"]})  # fmt: skip
    return pd.DataFrame(rows).drop_duplicates(["name", "start", "end"])


def summary(d: pd.DataFrame) -> dict:
    lab = d.dropna(subset=["y"])
    flags = d[d["risk"] >= BALANCED]
    return {"entities_per_text": round(len(d) / d["name"].nunique(), 1),
            "flags_per_text": round(len(flags) / d["name"].nunique(), 2), "flags": len(flags),
            "judge_false_found": int(lab["y"].sum()),
            "auroc": round(auroc(lab["y"].to_numpy(), lab["risk"].to_numpy()), 4),
            "flags_without_judge_label": round(float(flags["y"].isna().mean()), 3)}  # fmt: skip


if __name__ == "__main__":
    auto = pd.read_parquet(Path(sys.argv[1]) / "auto.parquet")
    texts = {}
    for f in sorted(Path(sys.argv[2]).rglob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            texts[r["name"]] = r["bio"]
    auto["text"] = [texts[r.name][r.start : r.end] for r in auto.itertuples()]  # pyright: ignore
    auto["y"] = auto["labels"].map(tuple).map({("Not Supported",): 1.0, ("Supported",): 0.0})
    auto = auto[~auto["text"].str.strip().str.lower().isin(PRONOUNS)]  # as the library does now
    auto = auto.merge(kinds(sorted(auto["name"].unique()), texts), on=["name", "start", "end"])
    auto["keep"] = [entities.is_specific(r.text, r.kind) for r in auto.itertuples()]  # pyright: ignore
    out = {}
    for kind, d in auto.groupby("set"):
        before, after = summary(d), summary(d[d["keep"]])
        after["flags_kept"] = round(after["flags"] / before["flags"], 3)
        after["judge_false_kept"] = round(
            after["judge_false_found"] / before["judge_false_found"], 3
        )
        out[str(kind)] = {"all": before, "specific": after}
    print(json.dumps(out, indent=1))
