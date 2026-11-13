"""SHARS at matched operating points (EVAL_PROTOCOL.md, follow-ups A).

Each labels directory holds one folder per arm with bios.jsonl (SHARS's output and log) and the
judge's results.jsonl: `original` = each arm judged against passages chosen by its own text (the
pre-registered labels), `shared` = every arm judged against one evidence set per person.

Run: uv run python -m hallustream.research.shars_followup <original_dir> <shared_dir> <out.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from hallustream.research.shars_test import _json_default, arm_table, paired, report

ARMS = ["naive", "halluse", "halluse_07", "probe_30", "probe_45", "probe_60", "logprob_45",
        "probe_v1"]  # fmt: skip
V3_MATCH = ["probe_30", "probe_45"]  # θ = 0.7544 and 0.5988: the candidates for matched coverage
# SHARS test people who are subjects of questions in the probe's training data
SEEN = {"Roberto Clemente", "Harrison Ford", "Errol Flynn", "Ronaldo (Brazilian footballer)"}
COLS = ("unsupported", "supported", "precision", "specific_unsupported", "specific_supported")


def both_answered(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired differences over the people both arms answered."""
    keep = set(a.loc[a["responded"], "name"]) & set(b.loc[b["responded"], "name"])
    a, b = a[a["name"].isin(keep)], b[b["name"].isin(keep)]
    out = {c: paired(a, b, c, n_boot, seed) for c in COLS}
    out["means"] = {c: [round(float(a[c].mean()), 3), round(float(b[c].mean()), 3)]
                    for c in ("unsupported", "supported", "precision")}  # fmt: skip
    return out


def per_answered(df: pd.DataFrame) -> dict:
    ans = df[df["responded"]]
    return {"unsupported": round(float(ans["unsupported"].mean()), 2),
            "supported": round(float(ans["supported"].mean()), 2)}  # fmt: skip


def analyze(root: Path) -> dict:
    tables = {a: arm_table(root / a) for a in ARMS if (root / a / "results.jsonl").exists()}
    rep = report(tables)
    for a, df in tables.items():
        pp = rep["arms"][a]["log"].get("per_person", {})
        if pp.get("props"):
            rep["arms"][a]["flag_rate"] = round(pp["props_flagged"] / pp["props"], 3)
        rep["arms"][a]["per_answered"] = per_answered(df)
    hs = tables["halluse"]
    matched = min(
        (a for a in V3_MATCH if a in tables),
        key=lambda a: abs(tables[a]["responded"].mean() - hs["responded"].mean()),
    )
    pairs = {f"{a}-halluse": (a, "halluse") for a in V3_MATCH + ["probe_60"] if a in tables}
    pairs |= {f"{a}-halluse_07": (a, "halluse_07") for a in V3_MATCH + ["probe_60"]
              if a in tables and "halluse_07" in tables}  # fmt: skip
    out = {"arms": rep["arms"], "matched_arm": matched, "pairs": {}}
    for key, (a, b) in pairs.items():
        ta, tb = tables[a], tables[b]
        not_seen = (ta[~ta["name"].isin(SEEN)], tb[~tb["name"].isin(SEEN)])
        out["pairs"][key] = {
            "all": {c: paired(ta, tb, c) for c in COLS},
            "both_answered": both_answered(ta, tb),
            "without_seen_people": {c: paired(*not_seen, c) for c in COLS},
        }
    m = out["pairs"][f"{matched}-halluse"]["all"]
    out["claim_fewer_false_at_matched_coverage"] = bool(
        m["unsupported"]["ci"][1] < 0
        and tables[matched]["supported"].mean() >= 0.9 * hs["supported"].mean()
    )
    out["response_rates"] = {a: round(float(t["responded"].mean()), 4) for a, t in tables.items()}
    return out


def curve(res: dict) -> list[dict]:
    """False against true claims per person, for the trade-off chart."""
    return [{"arm": a, "true": v["supported_per_bio"], "false": v["unsupported_per_bio"],
             "answered": v["response_rate"], "flag_rate": v.get("flag_rate")}
            for a, v in res["arms"].items()]  # fmt: skip


if __name__ == "__main__":
    res = {k: analyze(Path(p)) for k, p in (("original", sys.argv[1]), ("shared", sys.argv[2]))}
    for k in res:
        res[k]["curve"] = curve(res[k])
    Path(sys.argv[3]).write_text(json.dumps(res, indent=1, default=_json_default))
    for k, r in res.items():
        print(k, "matched:", r["matched_arm"], "claim:", r["claim_fewer_false_at_matched_coverage"])
        print("  ", json.dumps(r["pairs"][f"{r['matched_arm']}-halluse"]["all"]["unsupported"]))
