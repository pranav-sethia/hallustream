"""The project page (docs/index.html, served by GitHub Pages): data from the results, put into
site_template.html.

Inputs (all in results/, not committed):
  site/trace.json      per-token probe readings and library entities for the two live examples
                       (Modal run of the shipped library at d4e187c)
  site/showcase.json   the recorded stream() run of the Kiara Advani example, with timing
  site/topics_judge.jsonl  the fact checker's labels for the topic texts
  shars/test60/        the SHARS test (arms, judge labels)
  reserved/, topics/scores/  held-out detector scores (threshold curves, calibration)

Run: uv run python -m hallustream.research.site results docs/index.html
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from hallustream.research.factcheck import combine, judge
from hallustream.research.factcheck.validate import read_results
from hallustream.text import sentences

V3 = Path(__file__).parents[3] / "configs" / "detector_v3.json"
BALANCED, STRICT = 0.737832, 0.823369
MS_PER_TOKEN = 62.5  # measured on an L4 with the probe and GLiNER (docs/LIBRARY.md)

# Kiara Advani checked by hand against her Wikipedia article (2026-10-26).
KIARA = {
    "businesswoman": (
        False,
        "Wikipedia describes her as an actress. The businessman is her father.",
    ),
    "May 12, 1993": (False, "She was born on 31 July 1991."),
    "Mumbai": (True, "She was born in Bombay (Mumbai)."),
    "Rajesh Advani": (False, "Her father is Jagdeep Advani."),
    "Sushmita Dev": (False, "Her mother is Genevieve Jaffrey, a teacher."),
    "actor and producer": (False, "Her only sibling is a younger brother, Mishaal, a musician."),
    "Aryan Advani": (False, "Her only sibling is a younger brother, Mishaal, a musician."),
    "2013": (False, "Her film debut was Fugly, in 2014."),
    "Dil Dhadakne Do": (False, "Her debut was Fugly (2014). She was not in Dil Dhadakne Do."),
    "2014": (False, "She was not in Bajirao Mastani, which came out in 2015."),
    "Bajirao Mastani": (False, "She was not in Bajirao Mastani."),
    "Rani": (False, "She was not in Bajirao Mastani."),
    "2017": (False, "Her 2017 film was Machine."),
    "2018": (None, "The sentence was cut off before it named the film."),
}
# People whose three answers have no contradictory labels for the same detail (Egon Krenz and
# Anna May Wong had one each, from separate judge calls), in the order shown.
EXPLORER = ["Mike Trivisonno", "Kyra Sedgwick", "Daniel Dubois (boxer)", "Taika Waititi"]
LOST = {"Taika Waititi"}  # shown as the case where the probe arm did worse


def _arm(d: Path) -> dict[str, dict]:
    """Each person's answer in one SHARS arm, with the fact checker's labeled entities."""
    results, _ = read_results(d / "results.jsonl")
    out = {}
    for line in (d / "bios.jsonl").read_text().splitlines():
        r = json.loads(line)
        name, text = r["name"], r["bio"].strip()
        item = {"text": text, "ents": [], "true": 0, "false": 0}
        if text and name in results:
            raw = judge._json(results[name])
            claims = judge.parse_bio(results[name])["claims"]
            item["true"] = sum(lab == "Supported" for _, lab in claims)
            item["false"] = len(claims) - item["true"]
            ents = [(e["span"], e["label"]) for e in raw.get("entities") or []
                    if isinstance(e, dict) and e.get("label") in judge.ENTITY_LABELS]  # fmt: skip
            notes = [e.get("note", "") for e in raw.get("entities") or [] if isinstance(e, dict)
                     and e.get("label") in judge.ENTITY_LABELS]  # fmt: skip
            anns, _ = judge.locate(text, ents)
            k = 0  # notes follow the judge's order; locate keeps it and skips what it can't find
            for a in anns:
                while k < len(ents) and ents[k] != (a["span"], a["label"]):
                    k += 1
                note = notes[k] if k < len(notes) else ""
                k += 1
                if a["label"] != "Insufficient":
                    item["ents"].append([a["index"], a["index"] + len(a["span"]),
                                         a["label"] == "Supported", note])  # fmt: skip
        out[name] = item
    return out


def _strip_markdown(text: str) -> tuple[str, list[int], list[tuple[int, int]]]:
    """Text without **, * and ### markers; old-to-new offsets; ranges of former headings."""
    keep, heads, out = [], [], []
    for line in text.splitlines(keepends=True):
        head = re.match(r"#{1,6} ", line)
        start = len(out)
        for i, ch in enumerate(line):
            gone = (head and i < head.end()) or ch == "*"
            keep.append(len(out))
            if not gone:
                out.append(ch)
        if head:
            heads.append((start, len(out) - (1 if line.endswith("\n") else 0)))
    keep.append(len(out))
    return "".join(out), keep, heads


def _example(item: dict, verdicts: dict, cut: str, times: list[float] | None, missed: list[str],
             ) -> dict:  # fmt: skip
    """One live example: clean text, tokens with probe readings and reveal times, entities
    with risk, when they are checked, and how they compare with the source."""
    text = item["text"][: item["text"].index(cut) + len(cut)]
    clean, m, heads = _strip_markdown(text)
    ends = [hi for _, hi in sentences(clean)]
    toks = [t for t in item["tokens"] if t[1] <= len(text)]
    tokens = [[m[s], m[e], p, round(times[i] if times else i * MS_PER_TOKEN / 1000, 3)]
              for i, (s, e, p) in enumerate(toks)]  # fmt: skip
    ents = []
    for e in item["entities"]:
        if e["end"] > len(text):
            continue
        lo, hi = m[e["start"]], m[e["end"]]
        v = verdicts.get((e["start"], clean[lo:hi])) or verdicts.get(clean[lo:hi])
        ents.append({"text": clean[lo:hi], "start": lo, "end": hi, "risk": round(e["risk"], 3),
                     "flagged": e["risk"] >= BALANCED,
                     "checked": next((x for x in ends if x >= hi), len(clean)),
                     "true": v and v[0], "note": v[1] if v else ""})  # fmt: skip
    for name in missed:  # wrong details the entity finder never picked up
        lo = clean.index(name)
        v = verdicts[name]
        ents.append({"text": name, "start": lo, "end": lo + len(name), "risk": None,
                     "flagged": False, "checked": next(x for x in ends if x >= lo + len(name)),
                     "true": v[0], "note": v[1]})  # fmt: skip
    return {"id": item["id"], "prompt": item["prompt"], "text": clean, "heads": heads, "ends": ends,
            "tokens": tokens, "entities": sorted(ents, key=lambda e: e["start"])}  # fmt: skip


def hero(root: Path) -> list[dict]:
    items = {x["id"]: x for x in json.loads((root / "site" / "trace.json").read_text())}
    # Kiara: real timing from the recorded stream (time of the event that completed each token)
    run = json.loads((root / "site" / "showcase.json").read_text())[0]
    pos, marks = 0, []
    for ev in run["events"]:
        pos += len(ev["text"])
        marks.append((pos, ev["t"]))
    times = [
        next((t for p, t in marks if p >= e), marks[-1][1]) for _, e, _ in items["kiara"]["tokens"]
    ]
    kiara = _example(items["kiara"], KIARA, "she starred in the romantic drama", times,
                     ["Dil Dhadakne Do"])  # fmt: skip
    # The novel: the fact checker's labels and notes (first mention of each labeled entity)
    res, _ = read_results(root / "site" / "topics_judge.jsonl")
    raw = judge._json(res["A Caribbean Mystery"])
    found = [(e["span"], e["label"]) for e in raw["entities"]]
    notes = {e["span"]: e.get("note", "") for e in raw["entities"]}
    anns, _ = judge.locate(items["caribbean"]["text"], found)
    verdicts = {(a["index"], a["span"]): (a["label"] == "Supported", notes[a["span"]])
                for a in anns if a["label"] in ("Supported", "Not Supported")}  # fmt: skip
    novel = _example(items["caribbean"], verdicts, "sparks a series of investigations.", None, [])
    return [novel, kiara]


def explorer(root: Path) -> dict:
    d = root / "shars" / "test60"
    report = json.loads((d / "report.json").read_text())["arms"]
    keys = [("naive", "Qwen3-4B alone"), ("halluse", "SHARS + its sampling detector"),
            ("probe_45", "SHARS + HalluStream probe")]  # fmt: skip
    arms = {k: _arm(d / k) for k, _ in keys}
    people = [{"name": n.split(" (")[0], "lost": n in LOST,
               "arms": [{"text": arms[k][n]["text"], "true": arms[k][n]["true"],
                         "false": arms[k][n]["false"], "ents": arms[k][n]["ents"]}
                        for k, _ in keys]} for n in EXPLORER]  # fmt: skip
    summary = {k: report[k] for k in ("naive", "halluse", "probe_45", "probe_60", "logprob_45")}
    return {"labels": [label for _, label in keys], "people": people, "summary": summary}


def _risks(d: Path) -> tuple[np.ndarray, np.ndarray]:
    spans, hidden = combine.load(d)
    z = combine.apply(json.loads(V3.read_text())["detector"], spans, hidden)
    return 1 / (1 + np.exp(-z)), (spans["kind"] != "supported").to_numpy()


def curves(root: Path) -> dict:
    out = {}
    for name, d in [("people", root / "reserved"), ("topics", root / "topics" / "scores")]:
        r, y = _risks(d)
        pts = []
        for t in np.round(np.arange(0.30, 0.96, 0.01), 2):
            f = r >= t
            pts.append([float(t), round(float(f[y].mean()), 3), round(float(f[~y].mean()), 3),
                        round(float(y[f].mean()), 3)])  # fmt: skip
        idx = np.minimum((r * 10).astype(int), 9)
        cal = [[round(float(r[idx == b].mean()), 3), round(float(y[idx == b].mean()), 3),
                int((idx == b).sum())] for b in range(10)]  # fmt: skip
        out[name] = {"curve": pts, "calibration": cal, "entities": len(r)}
    return out


def build(root: Path) -> str:
    data = {"hero": hero(root), "shars": explorer(root), "thresholds": curves(root),
            "presets": {"balanced": BALANCED, "strict": STRICT}}  # fmt: skip
    page = (Path(__file__).parent / "site_template.html").read_text()
    return page.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))


if __name__ == "__main__":
    out = Path(sys.argv[2])
    out.write_text(build(Path(sys.argv[1])))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
