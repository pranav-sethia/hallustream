"""A static page with every SHARS test biography side by side, the judge's labels highlighted.

Run: uv run python -m hallustream.research.demo_page results/shars/test60 docs/demo/index.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hallustream.research.factcheck import judge
from hallustream.research.factcheck.validate import read_results

ARMS = [("naive", "Plain writing"), ("halluse", "SHARS + its own detector"),
        ("probe_45", "SHARS + probe")]  # fmt: skip
FIRST = "Mike Trivisonno"


def _arm(d: Path) -> dict[str, dict]:
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


def build(root: Path) -> str:
    arms = {key: _arm(root / key) for key, _ in ARMS}
    report = json.loads((root / "report.json").read_text())["arms"]
    names = sorted(arms["naive"], key=lambda n: (n != FIRST, n))
    people = [{"name": n, **{k: arms[k][n] for k, _ in ARMS}} for n in names]
    keys = ("supported_per_bio", "unsupported_per_bio", "precision", "response_rate")
    summary = [{"label": label, "detect": report[k].get("detect_seconds_per_prop"),
                **{m: report[k][m] for m in keys}} for k, label in ARMS]  # fmt: skip
    data = json.dumps({"arms": [list(a) for a in ARMS], "people": people, "summary": summary})
    page = (Path(__file__).parent / "demo_template.html").read_text()
    return page.replace("__DATA__", data.replace("</", "<\\/"))


if __name__ == "__main__":
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(Path(sys.argv[1])))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
