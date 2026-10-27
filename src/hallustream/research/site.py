"""The project page (docs/index.html, served by GitHub Pages): data from the results, put into
site_template.html.

Inputs (all in results/, not committed):
  site/trace.json      per-token probe readings and library entities for the two live examples
                       (Modal run of the shipped library at d4e187c)
  site/showcase.json   the recorded stream() run of the Kiara Advani example, with timing
  site/topics_judge.jsonl  the fact checker's labels for the topic texts
  shars/test182/       the SHARS test on all 182 people (arms, judge labels)
  paper_h2h.json       the head-to-head with the paper's Qwen2.5-7B probe
  reserved/, topics/scores/  held-out detector scores (threshold curves, calibration)

Run: uv run python -m hallustream.research.site results docs/index.html
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hallustream.entities import is_specific
from hallustream.research.factcheck import combine, judge
from hallustream.research.factcheck.validate import read_results
from hallustream.text import PRONOUNS, sentences

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
# Picked from all 182 by what the reader sees: the names, places, dates and titles the fact checker
# matched in each answer. Rule: the probe's answer has none made up and at least 4 verified, while
# both other answers keep made-up ones; then read in full and checked on Wikipedia. Earlier picks
# were dropped because their claim counts included vague praise that is not marked in the text.
SPOTLIGHT = "Adil Rami"
SPOTLIGHT_TRUTH = (
    "Adil Rami (born 27 December 1985 in Bastia) is a French former centre-back. He played for "
    "Lille, Valencia, AC Milan, Sevilla and Marseille, and won the 2018 World Cup with France."
)  # from his Wikipedia article, checked 2026-10-27
SPOTLIGHT_LEAD = (
    "Qwen3-4B half-knows him: it gets a French defender right, then invents a birth date, a "
    "birthplace and five clubs he never played for. SHARS with its own detector still keeps three "
    "of the fake clubs. With the probe, nothing made up is left."
)
# From the SHARS log of this run: the flagged propositions and their risks.
SPOTLIGHT_WIN = (
    "The probe flagged the fake birth date (98% risk), birthplace (93%) and clubs AS Monaco, PSG, "
    "Arsenal and Tottenham (60–95%). It also flagged Lille, his real first club, so that true "
    "detail was dropped too."
)
# Plain answers shown with the library's flags: test people with the most errors caught and no
# false alarms (pick_reader in the site notes), avoiding sensitive topics.
READER = ["Jimmy Bartel", "Teresa Palmer", "Chris Cheney", "Elsa Pataky"]


def _judged(text: str, result: str) -> list[list]:
    """The fact checker's labeled entities in `text`: [start, end, supported, note]."""
    raw = judge._json(result)
    labeled = [e for e in raw.get("entities") or []
               if isinstance(e, dict) and e.get("label") in judge.ENTITY_LABELS]  # fmt: skip
    ents = [(e["span"], e["label"]) for e in labeled]
    anns, _ = judge.locate(text, ents)
    out, k = [], 0  # notes follow the judge's order; locate keeps it and skips what it can't find
    for a in anns:
        while k < len(ents) and ents[k] != (a["span"], a["label"]):
            k += 1
        note = labeled[k].get("note", "") if k < len(labeled) else ""
        k += 1
        if a["label"] != "Insufficient":
            out.append([a["index"], a["index"] + len(a["span"]), a["label"] == "Supported", note])
    return out


def _arm(d: Path) -> dict[str, dict]:
    """Each person's answer in one SHARS arm, with the fact checker's labeled entities."""
    results, _ = read_results(d / "results.jsonl")
    out = {}
    for line in (d / "bios.jsonl").read_text().splitlines():
        r = json.loads(line)
        name, text = r["name"], r["bio"].strip()
        item = {"text": text, "ents": [], "true": 0, "false": 0}
        if text and name in results:
            claims = judge.parse_bio(results[name])["claims"]
            item["true"] = sum(lab == "Supported" for _, lab in claims)
            item["false"] = len(claims) - item["true"]
            item["ents"] = _judged(text, results[name])
        out[name] = item
    return out


def _shown(item: dict, name: str) -> dict:
    """Keep the marked details other than the person's own name, and count exactly those
    (distinct details), so the numbers above each answer match what is marked in it."""
    own = set(re.findall(r"\w+", name.lower()))
    ents = [e for e in item["ents"]
            if not set(re.findall(r"\w+", item["text"][e[0] : e[1]].lower())) <= own]  # fmt: skip
    ok = {item["text"][e[0] : e[1]].lower() for e in ents if e[2]}
    bad = {item["text"][e[0] : e[1]].lower() for e in ents if not e[2]}
    return item | {"ents": ents, "true": len(ok), "false": len(bad)}


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


def shars(root: Path) -> dict:
    """SHARS test numbers, and the spotlight person's three answers."""
    d = root / "shars" / "test182"
    report = json.loads((d / "report.json").read_text())
    keys = [("naive", "Qwen3-4B alone"), ("halluse", "SHARS + its sampling detector"),
            ("probe_45", "SHARS + HalluStream probe")]  # fmt: skip
    arms = {k: _shown(_arm(d / k)[SPOTLIGHT], SPOTLIGHT) for k, _ in keys}
    spotlight = {"name": SPOTLIGHT, "truth": SPOTLIGHT_TRUTH, "lead": SPOTLIGHT_LEAD,
                 "win": SPOTLIGHT_WIN,
                 "arms": [{"label": label, **arms[k]} for k, label in keys]}  # fmt: skip
    summary = {k: report["arms"][k] for k in ("naive", "halluse", "probe_45", "probe_60",
                                              "logprob_45", "probe_v1")}  # fmt: skip
    return {"summary": summary, "paired": report["paired"], "spotlight": spotlight}


def reader(root: Path) -> list[dict]:
    """Qwen3-4B's plain answers for test people, with the shipped library's flags (from the
    library check) and the fact checker's verdicts."""
    a = pd.read_parquet(root / "site" / "auto_kinds.parquet")
    a = a[a["set"] == "biographies"].drop_duplicates(["name", "start", "end"])
    a = a[[is_specific(t, k) and t.strip().lower() not in PRONOUNS
           for t, k in zip(a["ent"], a["kind"], strict=True)]]  # fmt: skip
    d = root / "shars" / "test60" / "naive"
    results, _ = read_results(d / "results.jsonl")
    bios = {
        (r := json.loads(x))["name"]: r["bio"] for x in (d / "bios.jsonl").read_text().splitlines()
    }
    out = []
    for name in READER:
        text = bios[name].rstrip()
        own = set(re.findall(r"\w+", name.lower()))
        is_name = lambda t, own=own: set(re.findall(r"\w+", t.lower())) <= own  # noqa: E731
        # The subject's own name is not a claim (the checker labels it by the claim around it)
        judged = [j for j in _judged(text, results[name]) if not is_name(text[j[0] : j[1]])]
        over = lambda lo, hi, j: j[0] < hi and lo < j[1]  # noqa: E731
        ents = []
        for r in a[a["name"] == name].sort_values("start").itertuples():
            lo, hi = int(r.start), int(r.end)  # pyright: ignore[reportArgumentType]
            if is_name(text[lo:hi]):
                continue
            hits = [j for j in judged if over(lo, hi, j)]
            wrong = [j for j in hits if not j[2]]
            true = False if wrong else (True if hits else None)
            note = wrong[0][3] if wrong else (hits[0][3] if hits else "")
            risk = float(r.risk)  # pyright: ignore[reportArgumentType]
            ents.append({"text": text[lo:hi], "start": lo, "end": hi, "risk": round(risk, 3),
                         "flagged": risk >= BALANCED, "true": true, "note": note})  # fmt: skip
        for j in judged:  # errors no detail was found for
            if not j[2] and not any(over(e["start"], e["end"], j) for e in ents):
                ents.append({"text": text[j[0]:j[1]], "start": j[0], "end": j[1], "risk": None,
                             "flagged": False, "true": False, "note": j[3]})  # fmt: skip
        errors = [j for j in judged if not j[2]]
        caught = sum(
            any(e["flagged"] and over(e["start"], e["end"], j) for e in ents) for j in errors
        )
        alarms = sum(e["flagged"] and e["true"] is True for e in ents)
        out.append({"name": name, "text": text, "entities": sorted(ents, key=lambda e: e["start"]),
                    "errors": len(errors), "caught": caught, "alarms": alarms})  # fmt: skip
    return out


def detectors(root: Path) -> dict:
    """Detectors on the same 987 SHARS development facts, and the every-layer sweep."""
    dev = json.loads((root / "shars" / "dev_choice_dedup.json").read_text())
    au = dev["auroc"]
    sweep = json.loads((root / "sweep_report.json").read_text())["Qwen3-4B"]
    layers = sorted(
        (r["layers"][0], r["val_auroc"]) for r in sweep["table"] if len(r["layers"]) == 1
    )
    return {"facts": dev["props"],
            "rows": [["HalluStream probe", au["probe_native"]],
                     ["Model's own confidence", max(au["surprise_shars"], au["surprise_native"])],
                     ["SHARS's sampling detector, best settings", au["halluse_exp_entail_a5"]],
                     ["SHARS's sampling detector, default settings", au["halluse_a3"]]],
            "sweep": layers, "sweep_baseline": sweep["baselines"]["entropy_max"],
            "sweep_spans": sweep["n_val_spans"]}  # fmt: skip


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


def paper(root: Path) -> dict:
    """Our probe vs the paper's released Qwen2.5-7B probe on the paper's own test sets."""
    h = json.loads((root / "paper_h2h.json").read_text())
    return {ds: {"paper": h[ds]["paper"]["auroc"], "ours": h[ds]["qwen25_L18"]["auroc"],
                 "gain": h[ds]["qwen25_L18_minus_paper"]["delta"]}
            for ds in ("longfact", "longfact_aug")}  # fmt: skip


def build(root: Path) -> str:
    data = {"hero": hero(root), "shars": shars(root), "reader": reader(root), "paper": paper(root),
            "detectors": detectors(root), "thresholds": curves(root),
            "presets": {"balanced": BALANCED, "strict": STRICT}}  # fmt: skip
    page = (Path(__file__).parent / "site_template.html").read_text()
    return page.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))


if __name__ == "__main__":
    out = Path(sys.argv[2])
    out.write_text(build(Path(sys.argv[1])))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
