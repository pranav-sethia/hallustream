"""Blind cross-check of the fact checker on the SHARS test by a model from another family.

`sample` draws 25 Supported and 25 Not Supported claims per arm (seed 7), shuffles them, and
writes batches with only (id, person, claim): no arm, no label. Each batch was labeled by a
Claude agent against the person's full English Wikipedia article (Supported if the article
states or clearly implies the claim). `compare` reports agreement with Gemini overall and per
arm.

Run (module hallustream.research.cross_judge):
  uv run python -m <module> sample results/shars/test60 results/audit_claude
  uv run python -m <module> compare results/audit_claude
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from hallustream.research.factcheck import judge
from hallustream.research.factcheck.validate import read_results

ARMS = ["naive", "halluse", "probe_45"]


def sample(root: Path, out: Path, per_label: int = 25, seed: int = 7) -> None:
    rng = random.Random(seed)
    items = []
    for arm in ARMS:
        res, _ = read_results(root / arm / "results.jsonl")
        pool: dict[str, list] = {"Supported": [], "Not Supported": []}
        for line in (root / arm / "bios.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["name"] in res and r["bio"].strip():
                for claim, lab in judge.parse_bio(res[r["name"]])["claims"]:
                    pool["Supported" if lab == "Supported" else "Not Supported"].append(
                        (r["name"], claim)
                    )
        for lab, claims in pool.items():
            for name, claim in rng.sample(claims, min(per_label, len(claims))):
                items.append({"person": name, "claim": claim, "arm": arm, "judge": lab})
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["id"] = i
    out.mkdir(parents=True, exist_ok=True)
    (out / "key.json").write_text(json.dumps(items, indent=1))
    people = sorted({it["person"] for it in items})
    for g in range(3):
        mine = set(people[g::3])
        blind = sorted(({"id": it["id"], "person": it["person"], "claim": it["claim"]}
                        for it in items if it["person"] in mine),
                       key=lambda x: (x["person"], x["id"]))  # fmt: skip
        (out / f"batch{g}.json").write_text(json.dumps(blind, indent=1))


def _kappa(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a == "Supported" for a, _ in pairs) / n
    pb = sum(b == "Supported" for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe)


def compare(d: Path) -> dict:
    key = {x["id"]: x for x in json.loads((d / "key.json").read_text())}
    lab = {x["id"]: x["label"] for f in sorted(d.glob("claude*.json"))
           for x in json.loads(f.read_text())}  # fmt: skip
    po, k = _kappa([(key[i]["judge"], lab[i]) for i in lab])
    out = {"claims": len(lab), "agreement": round(po, 3), "kappa": round(k, 3), "arms": {}}
    for arm in ARMS:
        pairs = [(key[i]["judge"], lab[i]) for i in lab if key[i]["arm"] == arm]
        out["arms"][arm] = {
            "agreement": round(_kappa(pairs)[0], 3),
            "gemini_supported_claude_not": pairs.count(("Supported", "Not Supported")),
            "gemini_not_claude_supported": pairs.count(("Not Supported", "Supported")),
        }
    return out


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        sample(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        print(json.dumps(compare(Path(sys.argv[2])), indent=1))
