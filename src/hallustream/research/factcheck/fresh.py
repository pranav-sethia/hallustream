"""The fresh test set of the 2026-11-13 follow-ups (EVAL_PROTOCOL.md, C).

600 people (150 per sitelink bucket) and 500 subjects that aren't people (25 per category x
bucket) from new random enwiki-20230401 titles (seed 2). Nobody and nothing used anywhere
before is eligible: FActScore's 500 + 183 people, the 1,200 people of configs/dev_people.json,
the 500 subjects of configs/topic_subjects.json, and anyone or anything named in the probe's
training, validation or test questions.

Run: uv run python -m hallustream.research.factcheck.fresh lookup <titles.json> <wikidata.json>
     uv run python -m hallustream.research.factcheck.fresh pick <wikidata.json> <factscore_data>
         configs/fresh_test.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hallustream.research.factcheck import people, topics

SEED = 2
PEOPLE_PER_BUCKET = 150
CONFIGS = Path(__file__).resolve().parents[4] / "configs"


def used_before(factscore_root: Path) -> set[str]:
    """Every person and subject an earlier check generated text about."""
    used = set(people.eval_people(factscore_root))
    used |= {b.name for b in people.human_labeled(factscore_root, "ChatGPT")}
    dev = json.loads((CONFIGS / "dev_people.json").read_text())
    used |= {p["name"] for p in dev["dev"] + dev["reserved"]}
    subjects = json.loads((CONFIGS / "topic_subjects.json").read_text())["subjects"]
    return used | {s["name"] for s in subjects}


def pick(wd: dict, factscore_root: Path) -> dict:
    from hallustream.research.data import load_records

    questions = [r.question for r in load_records()]
    text = "\n".join(questions).lower()
    used = used_before(factscore_root)
    humans = wd["humans"]
    exclude = used | {s for q in questions if (s := people.bio_subject(q))}
    exclude |= {n for n in humans if n.split(" (")[0].lower() in text}
    picked = people.stratified(humans, exclude, per_bucket=PEOPLE_PER_BUCKET, seed=SEED)
    items = {t: v for t, v in wd["items"].items() if t not in used}
    subjects = topics.pick(items, text, seed=SEED)
    names = {n for n, _, _ in picked} | {s["name"] for s in subjects}
    assert not names & used and len(names) == len(picked) + len(subjects)
    return {
        "note": "Fresh test set (EVAL_PROTOCOL.md, follow-ups C): random enwiki-20230401 titles "
        f"(seed {SEED}); people {PEOPLE_PER_BUCKET} per sitelink bucket "
        f"{people.SITELINK_BUCKETS}, subjects {topics.PER_CELL} per category x bucket; "
        "excludes every person and subject used before and anyone named in the probe's data",
        "titles_checked": wd["titles"],
        "humans_found": len(humans),
        "items_found": len(wd["items"]),
        "people": [{"name": n, "bucket": b, "sitelinks": k} for n, b, k in picked],
        "subjects": subjects,
    }


if __name__ == "__main__":
    if sys.argv[1] == "lookup":
        titles = json.loads(Path(sys.argv[2]).read_text())
        part = sys.argv[4] if len(sys.argv) > 4 else "both"
        out = {"titles": len(titles)}
        if part in ("both", "humans"):
            out["humans"] = people.wikidata_humans(titles)
        if part in ("both", "items"):
            out["items"] = topics.wikidata_items(titles)
        Path(sys.argv[3]).write_text(json.dumps(out))
        print({k: len(v) if isinstance(v, dict) else v for k, v in out.items()})
    else:
        wd = json.loads(Path(sys.argv[2]).read_text())
        res = pick(wd, Path(sys.argv[3]))
        Path(sys.argv[4]).write_text(json.dumps(res, indent=1))
        print(len(res["people"]), len(res["subjects"]))
