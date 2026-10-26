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


# Popularity buckets by Wikidata sitelink count (number of Wikipedia languages with a page).
SITELINK_BUCKETS = ((1, 3), (4, 10), (11, 30), (31, None))
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
USER_AGENT = "hallustream-research/0.1 (github.com/pranav-sethia/hallustream)"


def bucket(sitelinks: int) -> int:
    for i, (lo, hi) in enumerate(SITELINK_BUCKETS):
        if sitelinks >= lo and (hi is None or sitelinks <= hi):
            return i
    raise ValueError(sitelinks)


def sparql(query: str) -> list[dict]:
    """Rows of a Wikidata SPARQL query, retried on network errors."""
    import time
    import urllib.parse
    import urllib.request

    req = urllib.request.Request(
        WIKIDATA_SPARQL + "?" + urllib.parse.urlencode({"query": query, "format": "json"}),
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)["results"]["bindings"]
        except OSError:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("Wikidata query kept failing")


def wikidata_humans(titles: list[str], batch: int = 150) -> dict[str, int]:
    """English Wikipedia titles that are humans (P31 = Q5) -> sitelink count."""
    import time

    out: dict[str, int] = {}
    for i in range(0, len(titles), batch):
        values = " ".join(json.dumps(t) + "@en" for t in titles[i : i + batch])
        query = f"""SELECT ?title ?n WHERE {{
          VALUES ?title {{ {values} }}
          ?article schema:name ?title; schema:isPartOf <https://en.wikipedia.org/>;
                   schema:about ?item.
          ?item wdt:P31 wd:Q5; wikibase:sitelinks ?n.
        }}"""
        for row in sparql(query):
            out[row["title"]["value"]] = int(row["n"]["value"])
        time.sleep(1)
    return out


def stratified(humans: dict[str, int], exclude: set[str], per_bucket: int, seed: int = 0):
    """per_bucket people from each popularity bucket, none in `exclude`, in a fixed random order."""
    import random

    rng = random.Random(seed)
    picked = []
    for b in range(len(SITELINK_BUCKETS)):
        pool = sorted(n for n, k in humans.items() if bucket(k) == b and n not in exclude)
        if len(pool) < per_bucket:
            raise ValueError(f"bucket {b}: only {len(pool)} people")
        picked += [(n, b, humans[n]) for n in rng.sample(pool, per_bucket)]
    return picked


def bio_subject(question: str) -> str | None:
    """'Tell me a bio of X.' -> 'X' (training-data biography prompts)."""
    q = question.strip()
    if not q.startswith("Tell me a bio of "):
        return None
    return q[len("Tell me a bio of ") :].split("\n")[0].strip().rstrip(".")


if __name__ == "__main__":
    import sys

    # dev <humans.json> <factscore_data> <out.json>: 1,200 dev people, none used elsewhere
    from hallustream.research.data import load_records

    humans = json.loads(Path(sys.argv[2]).read_text())["humans"]
    root = Path(sys.argv[3])
    exclude = set(eval_people(root)) | {b.name for b in human_labeled(root, "ChatGPT")}
    questions = [r.question for r in load_records()]
    exclude |= {s for q in questions if (s := bio_subject(q))}
    text = "\n".join(questions).lower()  # also drop anyone named anywhere in the probe's data
    exclude |= {n for n in humans if n.split(" (")[0].lower() in text}
    picked = stratified(humans, exclude, per_bucket=300)
    dev = [p for i, p in enumerate(picked) if i % 300 < 150]
    reserved = [p for i, p in enumerate(picked) if i % 300 >= 150]
    assert not {n for n, _, _ in picked} & exclude
    Path(sys.argv[4]).write_text(json.dumps({
        "note": "Wikidata humans from random enwiki-20230401 pages, 300 per sitelink bucket "
                f"{SITELINK_BUCKETS}; excludes FActScore's 500 + 183 people and LongFact bios",
        "candidates_checked": len(humans), "excluded": len(exclude),
        "dev": [{"name": n, "bucket": b, "sitelinks": k} for n, b, k in dev],
        "reserved": [{"name": n, "bucket": b, "sitelinks": k} for n, b, k in reserved],
    }, indent=1))  # fmt: skip
    print(len(dev), len(reserved))
