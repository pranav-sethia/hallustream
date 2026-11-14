"""Summary of modal_jobs/library_check.py (docs/LIBRARY.md).

Run: uv run python -m hallustream.research.library_report <check dir> <scores dir of the parity
     texts> [detector config] (results/library_fresh results/fresh/topics; the first check:
     results/library results/topics/scores configs/detector_v3.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hallustream.research.factcheck import combine
from hallustream.research.metrics import auroc, paired_bootstrap

SHIPPED = Path(__file__).parents[3] / "configs" / "detector_v4.json"


def parity(lib: pd.DataFrame, scores: Path, config: Path = SHIPPED) -> dict:
    """Library scores vs the stored research scores of the same entities."""
    if "lib_v3" in lib:  # the first check's column name
        lib = lib.rename(columns={"lib_v3": "lib_adapted"})
    spans = pd.read_parquet(scores / "spans.parquet")
    hidden = np.load(scores / "hidden.npy").astype(np.float32)
    spans["ref_v1"] = combine.v1_logit(spans)
    spans["ref"] = combine.apply(json.loads(config.read_text())["detector"], spans, hidden)
    m = lib.merge(spans[["name", "start", "end", "ref_v1", "ref"]], on=["name", "start", "end"])
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    risk_diff = (sig(m["lib_adapted"]) - sig(m["ref"])).abs().max()
    return {"entities": len(lib), "matched": len(m), "texts": int(m["name"].nunique()),
            "max_abs_logit_diff": round(float((m["lib_adapted"] - m["ref"]).abs().max()), 5),
            "max_abs_risk_diff": round(float(risk_diff), 6)}  # fmt: skip


def automatic(auto: pd.DataFrame, coverage: pd.DataFrame) -> dict:
    """AUROC of GLiNER-found entities whose overlapping judge labels agree, by set."""
    out = {}
    for kind, d in auto.groupby("set"):
        labels = d["labels"].map(tuple)
        keep = labels.isin([("Not Supported",), ("Supported",)])
        d = d[keep]
        y = (labels[keep] == ("Not Supported",)).to_numpy().astype(int)
        risk = d["risk"].to_numpy()
        ci = paired_bootstrap(y, risk, np.full_like(risk, 0.5), d["name"].to_numpy(), n_boot=2000)
        cov = coverage[coverage["set"] == kind]
        found = cov.groupby("label")["found"].mean()
        out[str(kind)] = {
            "texts": int(auto[auto["set"] == kind]["name"].nunique()),
            "entities_found": int((auto["set"] == kind).sum()),
            "labeled": int(keep.sum()), "not_supported_share": round(float(y.mean()), 3),
            "auroc": round(auroc(y, risk), 4),
            "ci": [round(ci["ci_low"] + 0.5, 4), round(ci["ci_high"] + 0.5, 4)],
            "judge_false_entities_found": round(float(found.get("Not Supported", np.nan)), 3),
            "judge_true_entities_found": round(float(found.get("Supported", np.nan)), 3),
        }  # fmt: skip
    return out


if __name__ == "__main__":
    d, scores = Path(sys.argv[1]), Path(sys.argv[2])
    config = Path(sys.argv[3]) if len(sys.argv) > 3 else SHIPPED
    speed = json.loads((d / "speed.json").read_text())
    base = speed["plain"]["ms_per_token"]
    for v in speed.values():
        v["overhead"] = round(v["ms_per_token"] / base - 1, 4)
    print(json.dumps({
        "parity": parity(pd.read_parquet(d / "parity.parquet"), scores, config),
        "automatic": automatic(pd.read_parquet(d / "auto.parquet"),
                               pd.read_parquet(d / "coverage.parquet")),
        "speed": speed,
    }, indent=1))  # fmt: skip
