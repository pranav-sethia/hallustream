"""The public results bundle (results/public/) and every reported number recomputed from it.

The bundle holds only what this project made: per-entity labels and detector scores, Qwen3-4B's
texts, SHARS's per-person results and texts, and per-span scores for the head-to-head and the
base probe test (span ids and 0/1 labels only; the paper's annotations stay on Hugging Face).
No Wikipedia text and no judge notes (they quote Wikipedia).

Run: uv run python -m hallustream.research.public export     # from results/ (local, private)
     uv run python -m hallustream.research.public numbers    # results/public/numbers.json
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hallustream.research.metrics import auroc, paired_bootstrap, recall_at_fpr

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT / "results" / "public"
CONFIGS = ROOT / "configs"
# Own-answer groups: scores dir (spans.parquet + hidden.npy) and their role
OWN = {
    "dev": ("results/dev", "v3 and v4 training"),
    "gate": ("results/gate_v2", "v1 gate test (pre-registered), v4 training"),
    "reserved": ("results/reserved", "v3 test (pre-registered), v4 training"),
    "topics": ("results/topics/scores", "other-topics test (pre-registered), v4 training"),
    "fresh_people": ("results/fresh/people", "v4 test (follow-ups C)"),
    "fresh_topics": ("results/fresh/topics", "v4 test (follow-ups C)"),
}
SHARS = {"original": "results/shars/test182", "shared": "results/shars/shared182"}


def within_subject_auroc(y: np.ndarray, s: np.ndarray, groups: np.ndarray) -> float:
    """AUROC over (false, true) pairs from the same subject only: it leaves out how well a
    score tells apart subjects the model knows from ones it doesn't."""
    from scipy.stats import rankdata

    num = den = 0.0
    for g in np.unique(groups):
        m = groups == g
        yp, sp = y[m], s[m]
        npos, nneg = int(yp.sum()), int((1 - yp).sum())
        if npos and nneg:
            r = rankdata(sp)
            num += r[yp == 1].sum() - npos * (npos + 1) / 2
            den += npos * nneg
    return float(num / den)


def _own_table() -> pd.DataFrame:
    from hallustream.research.factcheck import combine

    v3 = json.loads((CONFIGS / "detector_v3.json").read_text())["detector"]
    v4 = json.loads((CONFIGS / "detector_v4.json").read_text())["detector"]
    parts = []
    for group, (d, role) in OWN.items():
        if not (ROOT / d / "spans.parquet").exists():
            continue
        df, hidden = combine.load(ROOT / d)
        out = df[["name", "kind", "truncated", "n_tokens", "entropy_max", "nll_max"]].copy()
        out.insert(0, "group", group)
        out["v1"] = combine.v1_logit(df)
        out["v3"] = combine.apply(v3, df, hidden)
        out["v4"] = combine.apply(v4, df, hidden)
        out["role"] = role
        parts.append(out)
    t = pd.concat(parts, ignore_index=True)
    return t.round({"entropy_max": 5, "nll_max": 5, "v1": 5, "v3": 5, "v4": 5})


def export() -> None:
    from hallustream.research.paper_h2h import load as h2h_load
    from hallustream.research.shars_test import arm_table

    PUBLIC.mkdir(parents=True, exist_ok=True)
    _own_table().to_parquet(PUBLIC / "own_answers.parquet", index=False)
    rows, texts = [], []
    for labels, d in SHARS.items():
        if not (ROOT / d).is_dir():
            continue
        for arm_dir in sorted((ROOT / d).iterdir()):
            if not (arm_dir / "results.jsonl").exists():
                continue
            t = arm_table(arm_dir)
            keep = ["name", "responded", "claims", "supported", "unsupported", "precision",
                    "specific_supported", "specific_unsupported", "seconds",
                    "detect_seconds"]  # fmt: skip
            t = t[[c for c in keep if c in t]].assign(arm=arm_dir.name, labels=labels)
            rows.append(t)
            if labels == "original":
                bios = (arm_dir / "bios.jsonl").read_text().splitlines()
                texts += [{"arm": arm_dir.name, "name": (r := json.loads(x))["name"],
                           "text": r["bio"]} for x in bios]  # fmt: skip
    pd.concat(rows, ignore_index=True).to_parquet(PUBLIC / "shars_people.parquet", index=False)
    with gzip.open(PUBLIC / "shars_texts.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in texts)
    h2h = h2h_load(ROOT / "data/h2h/spans", ROOT / "data/h2h/probes", ROOT / "data/gate/probs.npz",
                   ROOT / "data/gate_lora/probs.npz")  # fmt: skip
    cols = [c for c in h2h if c not in ("start", "end")]
    h2h[cols].round(6).to_parquet(PUBLIC / "h2h_qwen25.parquet", index=False)
    print({p.name: f"{p.stat().st_size / 1e6:.2f} MB" for p in sorted(PUBLIC.iterdir())})


def _ci(d: dict) -> list[float]:
    return [round(d["ci_low"], 4), round(d["ci_high"], 4)]


def numbers() -> dict:
    """Every headline number, recomputed from the bundle alone."""
    out: dict = {}
    h = pd.read_parquet(PUBLIC / "h2h_qwen25.parquet")
    for ds, g in h.groupby("dataset"):
        y, grp = g["label"].to_numpy(), g["answer_id"].to_numpy()
        r = {c: {"auroc": round(auroc(y, g[c].to_numpy()), 4),
                 "recall_at_10pct_fpr": round(recall_at_fpr(y, g[c].to_numpy()), 4)}
             for c in ("paper", "lora", "qwen25_L18", "qwen25_L26")}  # fmt: skip
        for ref in ("paper", "lora"):
            d = paired_bootstrap(y, g["qwen25_L18"].to_numpy(), g[ref].to_numpy(), grp, n_boot=2000)
            r[f"ours_minus_{ref}"] = {"delta": round(d["delta"], 4), "ci": _ci(d)}
        r["spans"] = len(g)
        out[f"h2h_{ds}"] = r
    own = pd.read_parquet(PUBLIC / "own_answers.parquet")
    for group, g in own.groupby("group"):
        y = (g["kind"] == "not_supported").to_numpy().astype(int)
        grp = g["name"].to_numpy()
        r = {"subjects": int(g["name"].nunique()), "entities": len(g),
             "share_false": round(float(y.mean()), 3)}  # fmt: skip
        for c in ("v1", "v3", "v4", "entropy_max"):
            r[c] = {
                "auroc": round(auroc(y, g[c].to_numpy()), 4),
                "within_subject": round(within_subject_auroc(y, g[c].to_numpy(), grp), 4),
            }
        for a, b in (("v4", "v3"), ("v3", "v1"), ("v4", "entropy_max"), ("v3", "entropy_max")):
            d = paired_bootstrap(y, g[a].to_numpy(), g[b].to_numpy(), grp, n_boot=2000)
            r[f"{a}_minus_{b}"] = {"delta": round(d["delta"], 4), "ci": _ci(d)}
        out[f"own_{group}"] = r
    fresh = own[own["group"].str.startswith("fresh")]
    if len(fresh):
        y = (fresh["kind"] == "not_supported").to_numpy().astype(int)
        grp = (fresh["group"] + ":" + fresh["name"]).to_numpy()
        d = paired_bootstrap(y, fresh["v4"].to_numpy(), fresh["v3"].to_numpy(), grp, n_boot=2000)
        out["fresh_v4_minus_v3"] = {"delta": round(d["delta"], 4), "ci": _ci(d)}
    sp = pd.read_parquet(PUBLIC / "shars_people.parquet")
    for (labels, arm), g in sp.groupby(["labels", "arm"]):
        ans = g[g["responded"]]
        out.setdefault(f"shars_{labels}", {})[arm] = {
            "people": len(g), "answered": round(float(g["responded"].mean()), 4),
            "false_per_person": round(float(g["unsupported"].mean()), 2),
            "true_per_person": round(float(g["supported"].mean()), 2),
            "precision": round(float(ans["precision"].mean()), 4),
        }  # fmt: skip
    (PUBLIC / "numbers.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export()
    else:
        print(json.dumps(numbers(), indent=1)[:3000])
