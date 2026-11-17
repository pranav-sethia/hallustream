"""The project page (docs/index.html, served by GitHub Pages): data from the results, put into
site_template.html.

Inputs (all in results/, not committed):
  site/trace.json      per-token probe readings and library entities for the two live examples
                       (Modal run of the shipped library at d4e187c)
  site/showcase.json   the recorded stream() run of the Kiara Advani example, with timing
  site/topics_judge.jsonl  the fact checker's labels for the topic texts
  shars/followup.json  every SHARS setting with shared-evidence labels (follow-ups A)
  paper_h2h.json, paper_h2h_llama31.json  the paper's released probes against the same recipe
  fresh/people, fresh/topics  the fresh test set's scores (threshold curves, calibration)
  public/numbers.json, test_results.json, sweep_report.json  headline scores and the layer sweep

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

SHIPPED = Path(__file__).parents[3] / "configs" / "detector_v4.json"
PRESETS = json.loads(
    (Path(__file__).parents[1] / "weights" / "qwen3-4b" / "thresholds.json").read_text()
)
BALANCED, STRICT = PRESETS["balanced"]["threshold"], PRESETS["strict"]["threshold"]
# The replays (hero, reader) were recorded with detector v3 at its balanced threshold.
V3_BALANCED = 0.737832
MS_PER_TOKEN = 62.5  # the replays: measured on an L4 with the probe and GLiNER at recording

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
                     "flagged": e["risk"] >= V3_BALANCED,
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
    """Every SHARS setting (shared-evidence labels) and the end-to-end speed-up at the two
    matched pairs of settings."""
    fu = json.loads((root / "shars" / "followup.json").read_text())["shared"]
    keep = ("response_rate", "supported_per_bio", "unsupported_per_bio", "precision",
            "seconds_per_bio", "detect_seconds_per_prop")  # fmt: skip
    summary = {k: {c: v.get(c) for c in keep} for k, v in fu["arms"].items()}
    speed = sorted(summary[h]["seconds_per_bio"] / summary[p]["seconds_per_bio"]
                   for h, p in (("halluse", "probe_30"), ("halluse_07", "probe_45")))  # fmt: skip
    return {"summary": summary, "speedup": [round(x, 2) for x in speed]}


def sweep(root: Path) -> dict:
    """The every-layer sweep on Qwen3-4B (validation AUROC per layer)."""
    s = json.loads((root / "sweep_report.json").read_text())["Qwen3-4B"]
    layers = sorted((r["layers"][0], r["val_auroc"]) for r in s["table"] if len(r["layers"]) == 1)
    return {"layers": layers, "baseline": s["baselines"]["entropy_max"], "spans": s["n_val_spans"]}


def _risks(d: Path) -> tuple[np.ndarray, np.ndarray]:
    spans, hidden = combine.load(d)
    z = combine.apply(json.loads(SHIPPED.read_text())["detector"], spans, hidden)
    return 1 / (1 + np.exp(-z)), (spans["kind"] != "supported").to_numpy()


def curves(root: Path) -> dict:
    """Threshold curves and calibration of the shipped detector on the fresh test set."""
    out = {}
    for name, d in [("people", root / "fresh" / "people"), ("topics", root / "fresh" / "topics")]:
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
    """The paper's released linear and LoRA probes and the same recipe with a middle layer and
    tuned training, per model and test set, all scored on the same details."""
    out = {}
    for model, key, f in (("Qwen2.5-7B", "qwen25_L18", "paper_h2h.json"),
                          ("Llama-3.1-8B", "llama31_L12", "paper_h2h_llama31.json")):  # fmt: skip
        h = json.loads((root / f).read_text())
        out[model] = {ds: {"linear": h[ds]["paper"]["auroc"], "lora": h[ds]["lora"]["auroc"],
                           "ours": h[ds][key]["auroc"], "reported": h[ds]["paper_reported"]}
                      for ds in ("longfact", "longfact_aug", "healthbench") if ds in h}  # fmt: skip
    return out


def own(root: Path) -> dict:
    """The shipped detector on the fresh test (from the public bundle's numbers) and the base
    probe on other models' text."""
    n = json.loads((root / "public" / "numbers.json").read_text())
    base = json.loads((root / "test_results.json").read_text())["models"]["qwen3_L21"]["overall"]
    fresh = {k: n[f"own_fresh_{k}"] for k in ("people", "topics")}
    out = {k: {"subjects": f["subjects"], "ours": f["v4"]["auroc"], "v1": f["v1"]["auroc"],
               "confidence": f["entropy_max"]["auroc"]} for k, f in fresh.items()}  # fmt: skip
    out["base"] = {"details": base["n_spans"], "ours": base["auroc"],
                   "confidence": base["baseline_auroc"]}  # fmt: skip
    return out


def build(root: Path) -> str:
    data = {"hero": hero(root), "shars": shars(root), "paper": paper(root), "sweep": sweep(root),
            "thresholds": curves(root), "own": own(root),
            "presets": {"balanced": BALANCED, "strict": STRICT}}  # fmt: skip
    page = (Path(__file__).parent / "site_template.html").read_text()
    return page.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))


if __name__ == "__main__":
    out = Path(sys.argv[2])
    out.write_text(build(Path(sys.argv[1])))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
