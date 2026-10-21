"""Judge checks against FActScore human labels, and the own-answer gate (EVAL_PROTOCOL.md).

Inputs are batch results files ({"key", "response"} JSONL) downloaded from the Volume.

Run: uv run python -m hallustream.research.factcheck.validate checks <factscore_data> <results>
     uv run python -m hallustream.research.factcheck.validate gate <spans.parquet> [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hallustream.research.factcheck import judge, people
from hallustream.research.metrics import auroc, paired_bootstrap

LABELING_BAR = 0.80
END_TO_END_BAR = 0.03
GATE_BAR = 0.75


def read_results(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        try:
            out[row["key"]] = row["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            continue
    return out


def balanced_accuracy(human: list[str], judged: list[str]) -> float:
    """Mean of recall on human-Supported and recall on human-Not-Supported claims."""
    h = np.array([x == "S" for x in human])
    j = np.array([x == "Supported" for x in judged])
    return float(((j & h).sum() / h.sum() + (~j & ~h).sum() / (~h).sum()) / 2)


def check_labeling(root: Path, results: dict[str, str]) -> dict:
    human, judged, failed = [], [], 0
    for system in people.LABELED_SYSTEMS:
        for b in people.human_labeled(root, system):
            text = results.get(f"{system}|{b.name}")
            if not b.claims:
                continue
            try:
                labels = judge.parse_claims(text, len(b.claims)) if text else [None] * len(b.claims)
            except (json.JSONDecodeError, AttributeError):
                labels = [None] * len(b.claims)
            for (_, h), j in zip(b.claims, labels, strict=True):
                if j is None:
                    failed += 1
                    continue
                human.append(h)
                judged.append(j)
    acc = balanced_accuracy(human, judged)
    return {"claims": len(human), "unparsed": failed, "balanced_accuracy": round(acc, 4),
            "passes": acc >= LABELING_BAR}  # fmt: skip


def check_end_to_end(root: Path, results: dict[str, str]) -> dict:
    """Per system: mean per-bio FActScore from the judge's own claims vs from the human claims."""
    out = {}
    for system in people.LABELED_SYSTEMS:
        hs, js = [], []
        for b in people.human_labeled(root, system):
            text = results.get(f"{system}|{b.name}")
            if b.factscore is None or not text:
                continue
            try:
                claims = judge.parse_bio(text)["claims"]
            except (json.JSONDecodeError, AttributeError):
                continue
            if not claims:
                continue
            hs.append(b.factscore)
            js.append(sum(lab == "Supported" for _, lab in claims) / len(claims))
        err = abs(float(np.mean(js)) - float(np.mean(hs)))
        out[system] = {"bios": len(hs), "human": round(float(np.mean(hs)), 4),
                       "judge": round(float(np.mean(js)), 4), "error": round(err, 4)}  # fmt: skip
    out["passes"] = all(v["error"] <= END_TO_END_BAR for k, v in out.items() if k != "passes")
    return out


def gate(spans: pd.DataFrame) -> dict:
    """Entity AUROC on the model's own biographies, bootstrapped over people."""
    res = {"people": int(spans["name"].nunique()), "kinds": spans["kind"].value_counts().to_dict()}
    for name, pos in (
        ("primary", {"not_supported"}),
        ("secondary", {"not_supported", "insufficient"}),
    ):
        d = spans if name == "secondary" else spans[spans["kind"] != "insufficient"]
        y = d["kind"].isin(pos).to_numpy().astype(int)
        groups = d["name"].to_numpy()
        probe, base = d["probe"].to_numpy(), d["nll_max"].to_numpy()
        a = auroc(y, probe)
        lo_hi = paired_bootstrap(y, probe, np.full_like(probe, 0.5), groups)  # AUROC(probe) - 0.5
        vs = paired_bootstrap(y, probe, base, groups)
        res[name] = {
            "entities": int(len(y)), "positives": int(y.sum()),
            "auroc": round(a, 4),
            "ci": [round(lo_hi["ci_low"] + 0.5, 4), round(lo_hi["ci_high"] + 0.5, 4)],
            "baseline_auroc": round(auroc(y, base), 4),
            "probe_minus_baseline": {k: round(v, 4) for k, v in vs.items()},
        }  # fmt: skip
    res["passes"] = res["primary"]["auroc"] >= GATE_BAR
    return res


if __name__ == "__main__":
    if sys.argv[1] == "checks":
        root, rdir = Path(sys.argv[2]), Path(sys.argv[3])
        out = {
            "labeling": check_labeling(root, read_results(rdir / "check_labeling.jsonl")),
            "end_to_end": check_end_to_end(root, read_results(rdir / "check_end_to_end.jsonl")),
        }
    else:
        out = {Path(p).parent.name: gate(pd.read_parquet(p)) for p in sys.argv[2:]}
    print(json.dumps(out, indent=1))
