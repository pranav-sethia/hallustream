"""Judge checks against FActScore human labels, and the own-answer gate (EVAL_PROTOCOL.md).

Inputs are batch results files ({"key", "response" | "error"} JSONL) downloaded from the Volume.
Check keys are "<prefix><system>|<name>", e.g. "labeling|ChatGPT|Ada Lovelace".

Run (module hallustream.research.factcheck.validate):
  checks <factscore_data> <results.jsonl> <key prefix> tune|report
  gate <checks.json> <gate_dir> [<gate_dir> ...]
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from hallustream.research.factcheck import judge, people
from hallustream.research.metrics import auroc, paired_bootstrap

LABELING_BAR = 0.80
END_TO_END_BAR = 0.03
GATE_BAR = 0.75
FROZEN = Path("configs/frozen_choices.json")
SEEN = Path("configs/bio_people_in_probe_data.json")  # people with training answers


def read_results(path: Path) -> tuple[dict[str, str], dict]:
    """key -> answer text, plus counts of errors and cut-off answers."""
    out: dict[str, str] = {}
    stats: Counter = Counter()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["key"] in out:
            raise ValueError(f"duplicate key in results: {row['key']}")
        text, finish = judge.response_text(row)
        stats["lines"] += 1
        stats[f"finish_{finish}"] += 1
        if text is not None:
            out[row["key"]] = text
    return out, dict(stats)


def balanced_accuracy(human: list[str], judged: list[str]) -> float:
    """Mean of recall on human-Supported and recall on human-Not-Supported claims."""
    h = np.array([x == "S" for x in human])
    j = np.array([x == "Supported" for x in judged])
    if h.all() or not h.any():
        return float("nan")
    return float(((j & h).sum() / h.sum() + (~j & ~h).sum() / (~h).sum()) / 2)


def _labeled(root: Path, subset: set[str] | None):
    for system in people.LABELED_SYSTEMS:
        for b in people.human_labeled(root, system):
            if subset is None or b.name in subset:
                yield system, b


def check_labeling(root: Path, results: dict[str, str], prefix: str, subset=None) -> dict:
    """Judge labels on the humans' own S/NS facts. A bio whose answer is missing,
    malformed or skips a claim counts as failed, with all its claims."""
    human, judged, bios, failed = [], [], 0, 0
    for system, b in _labeled(root, subset):
        if not b.claims:
            continue
        bios += 1
        try:
            labels = judge.parse_claims(results[f"{prefix}{system}|{b.name}"], len(b.claims))
        except (KeyError, ValueError):
            failed += 1
            continue
        human += [h for _, h in b.claims]
        judged += labels
    acc = balanced_accuracy(human, judged)
    return {"bios": bios, "failed_bios": failed, "claims": len(human),
            "balanced_accuracy": round(acc, 4),
            "passes": bool(acc >= LABELING_BAR)}  # fmt: skip


def check_end_to_end(root: Path, results: dict[str, str], prefix: str, subset=None) -> dict:
    """Per system: mean per-bio FActScore from the judge's own claims vs the human one.

    A bio the judge could not handle (missing, malformed, or no claims for a bio the
    humans found facts in) is reported as failed and left out of both means.
    """
    out = {}
    for system in people.LABELED_SYSTEMS:
        hs, js, n_judge, n_human, failed = [], [], [], [], 0
        for _, b in _labeled(root, subset):
            if b.system != system or b.factscore is None:
                continue
            try:
                claims = judge.parse_bio(results[f"{prefix}{system}|{b.name}"])["claims"]
            except (KeyError, ValueError):
                claims = []
            if not claims:
                failed += 1
                continue
            hs.append(b.factscore)
            js.append(sum(lab == "Supported" for _, lab in claims) / len(claims))
            n_judge.append(len(claims))
            n_human.append(len(b.facts))
        err = abs(float(np.mean(js)) - float(np.mean(hs)))
        out[system] = {"bios": len(hs), "failed_bios": failed,
                       "human": round(float(np.mean(hs)), 4), "judge": round(float(np.mean(js)), 4),
                       "error": round(err, 4),
                       "claims_per_bio": {"judge": round(float(np.mean(n_judge)), 1),
                                          "human": round(float(np.mean(n_human)), 1)}}  # fmt: skip
    out["passes"] = all(v["error"] <= END_TO_END_BAR for k, v in out.items() if k != "passes")
    return out


def _mentions(span: str, fact: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(span.lower()) + r"(?!\w)", fact.lower()) is not None


def check_entities(root: Path, results: dict[str, str], prefix: str, subset=None) -> dict:
    """Rough entity check (reported, no bar): an entity mentioned only in facts the humans
    marked S should be Supported; one mentioned only in NS facts, Not Supported."""
    human, judged = [], []
    for system, b in _labeled(root, subset):
        try:
            ents = judge.parse_bio(results[f"{prefix}{system}|{b.name}"])["entities"]
        except (KeyError, ValueError):
            continue
        for span, label in ents:
            if label == "Insufficient":
                continue
            labs = {lab for text, lab in b.claims if _mentions(span, text)}
            if len(labs) == 1:
                human.append(labs.pop())
                judged.append(label)
    return {"entities": len(human), "human_NS": human.count("NS"),
            "balanced_accuracy": round(balanced_accuracy(human, judged), 4)}  # fmt: skip


def checks(root: Path, results: dict[str, str], prefix: str, subset=None) -> dict:
    out = {
        "labeling": check_labeling(root, results, f"{prefix}labeling|", subset),
        "end_to_end": check_end_to_end(root, results, f"{prefix}e2e|", subset),
        "entities": check_entities(root, results, f"{prefix}e2e|", subset),
    }
    out["passes"] = out["labeling"]["passes"] and out["end_to_end"]["passes"]
    return out


def gate(spans: pd.DataFrame, baseline: str, names: set[str] | None = None) -> dict:
    """Entity AUROC on the model's own biographies, bootstrapped over people.

    The probe score is the mean over the trained seeds (columns p_seed*).
    """
    if names is not None:
        spans = spans[spans["name"].isin(names)]
    seeds = sorted(c for c in spans.columns if c.startswith("p_seed"))
    spans = spans.assign(probe=spans[seeds].mean(axis=1))
    res = {"people": int(spans["name"].nunique()), "baseline": baseline,
           "kinds": spans["kind"].value_counts().to_dict()}  # fmt: skip
    for name, pos in (
        ("primary", {"not_supported"}),
        ("secondary", {"not_supported", "insufficient"}),
    ):
        d = spans if name == "secondary" else spans[spans["kind"] != "insufficient"]
        y = d["kind"].isin(pos).to_numpy().astype(int)
        groups = d["name"].to_numpy()
        probe, base = d["probe"].to_numpy(), d[baseline].to_numpy()
        half = np.full_like(probe, 0.5)
        ci = paired_bootstrap(y, probe, half, groups)  # AUROC(probe) - 0.5
        base_ci = paired_bootstrap(y, base, half, groups)
        res[name] = {
            "entities": int(len(y)), "positives": int(y.sum()),
            "auroc": round(auroc(y, probe), 4),
            "ci": [round(ci["ci_low"] + 0.5, 4), round(ci["ci_high"] + 0.5, 4)],
            "auroc_by_seed": {s: round(auroc(y, d[s].to_numpy()), 4) for s in seeds},
            "baseline_auroc": round(auroc(y, base), 4),
            "baseline_ci": [round(base_ci["ci_low"] + 0.5, 4), round(base_ci["ci_high"] + 0.5, 4)],
            "probe_minus_baseline": {k: round(v, 4) for k, v in
                                     paired_bootstrap(y, probe, base, groups).items()},
        }  # fmt: skip
    res["passes"] = res["primary"]["auroc"] >= GATE_BAR
    return res


def _gate_dirs(dirs: list[Path]) -> dict:
    frozen = json.loads(FROZEN.read_text())["models"]
    runs = {}
    for d in dirs:
        variant = json.loads((d / "run_meta.json").read_text())["config"]["probe"]
        runs[d.name] = (pd.read_parquet(d / "spans.parquet"), frozen[variant]["best_baseline"])
    out = {m: gate(df, base) for m, (df, base) in runs.items()}
    out["untruncated"] = {m: gate(df[~df["truncated"]], base) for m, (df, base) in runs.items()}
    seen = {n for n, c in json.loads(SEEN.read_text())["people"].items() if c.get("train")}
    for part, keep in (("seen_in_training", True), ("unseen_in_training", False)):
        out[part] = {m: gate(df, base, {n for n in set(df["name"]) if (n in seen) == keep})
                     for m, (df, base) in runs.items()}  # fmt: skip
    if len(runs) > 1:  # same people for every model, since response rates differ
        common = set.intersection(*(set(df["name"]) for df, _ in runs.values()))
        out["common_people"] = {m: gate(df, base, common) for m, (df, base) in runs.items()}
    return out


if __name__ == "__main__":
    if sys.argv[1] == "checks":
        root, path, prefix, part = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]
        tune, report = people.split_labeled([b.name for b in people.human_labeled(root, "ChatGPT")])
        results, stats = read_results(path)
        out = {"results": stats} | checks(root, results, prefix, tune if part == "tune" else report)
    else:
        status = json.loads(Path(sys.argv[2]).read_text())
        if not status.get("passes"):
            sys.exit("judge checks have not passed; the gate is not read (EVAL_PROTOCOL.md)")
        out = _gate_dirs([Path(p) for p in sys.argv[3:]])
    print(json.dumps(out, indent=1))
