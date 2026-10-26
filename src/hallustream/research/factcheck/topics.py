"""Subjects other than people, for checking the detector beyond biographies.

Random English Wikipedia pages whose Wikidata item is not a human, sorted into five broad
categories by the labels of their "instance of" classes, and sampled evenly by category and
popularity (sitelink bucket). See EVAL_PROTOCOL.md, "Other topics".

Run: uv run python -m hallustream.research.factcheck.topics pick <titles.json> <out.json>
     uv run python -m hallustream.research.factcheck.topics eval <scores dir> <subjects.json>
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from hallustream.research.factcheck.people import SITELINK_BUCKETS, bucket, sparql

TOPIC_PROMPT = "Tell me about {name}."

# Checked in this order; a subject takes the first category with a matching class label.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "living_thing": ("taxon", "species", "genus", "breed", "cultivar"),
    "event": ("battle", "war", "election", "tournament", "championship", "competition",
              "festival", "season", "conflict", "earthquake", "disaster", "olympic", "event",
              "race", "massacre", "siege", "referendum", "hurricane", "cyclone", "storm"),
    "organization": ("company", "business", "enterprise", "organization", "organisation",
                     "club", "team", "school", "university", "college", "political party",
                     "band", "association", "brand", "airline", "record label", "agency",
                     "institution", "union", "society", "foundation", "corporation"),
    "creative_work": ("film", "album", "single", "song", "book", "novel", "television",
                      "episode", "video game", "painting", "play", "musical", "poem",
                      "magazine", "newspaper", "anime", "manga", "comic", "periodical",
                      "opera", "series", "work"),
    "place": ("village", "town", "city", "municipality", "commune", "settlement", "county",
              "district", "province", "region", "river", "lake", "mountain", "island", "park",
              "neighborhood", "neighbourhood", "suburb", "hamlet", "census-designated place",
              "unincorporated community", "building", "church", "station", "bridge", "castle",
              "museum", "street", "road", "airport", "stadium", "dam", "valley", "bay", "hill",
              "parish", "locality", "temple", "fort", "house"),
}  # fmt: skip
PER_CELL = 25  # subjects per (category, sitelink bucket): 5 x 4 x 25 = 500


def category(classes: list[str]) -> str | None:
    """Broad category from "instance of" class labels; None for anything else."""
    low = [c.lower() for c in classes]
    for cat, words in CATEGORIES.items():
        if any(re.search(rf"\b{re.escape(w)}\b", c) for w in words for c in low):
            return cat
    return None


def wikidata_items(titles: list[str], batch: int = 150) -> dict[str, dict]:
    """English Wikipedia titles that are not humans -> sitelinks and "instance of" labels.
    Wikimedia pages (disambiguation, lists, categories, templates) are left out."""
    import time

    out: dict[str, dict] = {}
    for i in range(0, len(titles), batch):
        values = " ".join(json.dumps(t) + "@en" for t in titles[i : i + batch])
        query = f"""SELECT ?title ?n (GROUP_CONCAT(DISTINCT ?label; separator="|") AS ?classes)
        WHERE {{
          VALUES ?title {{ {values} }}
          ?article schema:name ?title; schema:isPartOf <https://en.wikipedia.org/>;
                   schema:about ?item.
          ?item wikibase:sitelinks ?n; wdt:P31 ?class.
          ?class rdfs:label ?label. FILTER(lang(?label) = "en")
          FILTER NOT EXISTS {{ ?item wdt:P31 wd:Q5 }}
        }} GROUP BY ?title ?n"""
        for row in sparql(query):
            classes = row["classes"]["value"].split("|")
            if any(c.startswith("Wikimedia") for c in classes):
                continue
            out[row["title"]["value"]] = {"sitelinks": int(row["n"]["value"]), "classes": classes}
        time.sleep(1)
    return out


def pick(items: dict[str, dict], exclude_text: str, per_cell: int = PER_CELL, seed: int = 0):
    """per_cell subjects from each (category, sitelink bucket), in a fixed random order.
    A subject is left out if its title (without a parenthetical) appears in exclude_text."""
    rng = random.Random(seed)
    picked = []
    for cat in CATEGORIES:
        for b in range(len(SITELINK_BUCKETS)):
            pool = sorted(
                t for t, v in items.items()
                if category(v["classes"]) == cat and bucket(v["sitelinks"]) == b
                and t.split(" (")[0].lower() not in exclude_text
            )  # fmt: skip
            if len(pool) < per_cell:
                raise ValueError(f"{cat}, bucket {b}: only {len(pool)} subjects")
            picked += [{"name": t, "category": cat, "bucket": b,
                        "sitelinks": items[t]["sitelinks"], "classes": items[t]["classes"]}
                       for t in rng.sample(pool, per_cell)]  # fmt: skip
    return picked


def evaluate(d: Path, subjects: list[dict], n_boot: int = 2000) -> dict:
    """Entity AUROC of v1, v3 and the model's confidence on the scored texts, with intervals
    from a bootstrap over subjects; overall and by category (Supported vs Not Supported)."""
    import numpy as np

    from hallustream.research.factcheck import combine
    from hallustream.research.metrics import auroc, paired_bootstrap

    df, hidden = combine.load(d)
    det = json.loads((Path(__file__).parents[4] / "configs" / "detector_v3.json").read_text())
    scores = {
        "v3": combine.apply(det["detector"], df, hidden),
        "v1": combine.v1_logit(df),
        "max_entropy": df["entropy_max"].to_numpy(),
        "max_surprise": df["nll_max"].to_numpy(),
    }
    y = (df["kind"] == "not_supported").to_numpy().astype(int)
    groups = df["name"].to_numpy()
    cats = {s["name"]: s["category"] for s in subjects}

    def block(mask: np.ndarray) -> dict:
        out = {"subjects": int(len(set(groups[mask]))), "entities": int(mask.sum()),
               "not_supported": round(float(y[mask].mean()), 3)}  # fmt: skip
        out |= {k: round(auroc(y[mask], v[mask]), 4) for k, v in scores.items()}
        return out

    res = {"all": block(np.ones(len(y), bool))}
    for a, b in (("v3", "max_entropy"), ("v1", "max_entropy"), ("v3", "v1")):
        ci = paired_bootstrap(y, scores[a], scores[b], groups, n_boot=n_boot)
        res[f"{a}-{b}"] = {k: round(v, 4) for k, v in ci.items()}
    cat = np.array([cats.get(n) for n in groups])
    res["by_category"] = {c: block(cat == c) for c in CATEGORIES if (cat == c).any()}
    return res


if __name__ == "__main__" and sys.argv[1] == "eval":
    subjects = json.loads(Path(sys.argv[3]).read_text())["subjects"]
    print(json.dumps(evaluate(Path(sys.argv[2]), subjects), indent=1))
elif __name__ == "__main__":
    from hallustream.research.data import load_records

    titles = json.loads(Path(sys.argv[2]).read_text())
    items = wikidata_items(titles)
    # Leave out any subject named in the probe's training, validation or test questions.
    text = "\n".join(r.question for r in load_records()).lower()
    picked = pick(items, text)
    Path(sys.argv[3]).write_text(json.dumps({
        "note": "Non-human Wikidata items from random enwiki-20230401 pages, "
                f"{PER_CELL} per category x sitelink bucket {SITELINK_BUCKETS}; "
                "excludes subjects named in the probe's data",
        "titles_checked": len(titles), "items": len(items), "subjects": picked,
    }, indent=1))  # fmt: skip
    print({"titles": len(titles), "items": len(items), "subjects": len(picked)})
