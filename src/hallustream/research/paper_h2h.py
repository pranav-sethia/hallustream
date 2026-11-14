"""Head-to-head with Obeso et al.'s released Qwen2.5-7B probe on its own test sets.

Our probes (mean of 5 seeds) and the paper's released probe are scored on exactly the same
test spans: each span's highest token probability, exact character offsets. The paper's
per-token probabilities come from the reproduction run (data/gate/probs.npz). Pre-registered in
EVAL_PROTOCOL.md ("Against the paper on its own benchmark").

The released LoRA probe (`qwen2_5_7b_lora_lambda_kl_0_05`, follow-ups B1) is added when its
per-token probabilities (modal_jobs/gate.py::lora) are given as a fourth argument.

Run: uv run python -m hallustream.research.paper_h2h <spans_dir> <probes_dir> data/gate/probs.npz
         [data/gate_lora/probs.npz] [--model=llama31]
     spans_dir: the main_qwen25 cache's spans_c*.parquet files (any layout below it)
     probes_dir: probes/<variant>/spans_test.parquet for each variant
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hallustream.research.metrics import auroc, paired_bootstrap, recall_at_fpr

# Per model: our variants (the validation-chosen layer first, then the paper's layer) and the
# paper's published AUROC for its released probes (linear; LoRA where released and reported).
MODELS = {
    "qwen25": {"variants": ["qwen25_L18", "qwen25_L26"],
               "reported": {"longfact": 0.8383, "longfact_aug": 0.8467},
               "reported_lora": {}},
    "llama31": {"variants": ["llama31_L12", "llama31_L30"],
                "reported": {"longfact": 0.8535, "longfact_aug": 0.8678, "healthbench": 0.8560},
                "reported_lora": {"longfact": 0.8938, "longfact_aug": 0.9036,
                                  "healthbench": 0.8960}},
}  # fmt: skip
PAPER_REPORTED = MODELS["qwen25"]["reported"]
VARIANTS = MODELS["qwen25"]["variants"]


def load(spans_dir: Path, probes_dir: Path, npz_path: Path, lora_npz: Path | None = None,
         model: str = "qwen25") -> pd.DataFrame:  # fmt: skip
    variants = MODELS[model]["variants"]
    spans = pd.concat([pd.read_parquet(f) for f in sorted(spans_dir.rglob("*spans_c*.parquet"))])
    spans = spans[spans["split"] == "test"][["answer_id", "span_id", "start", "end"]]
    df = None
    for v in variants:
        p = pd.read_parquet(probes_dir / v / "spans_test.parquet")
        seeds = [c for c in p if c.startswith("p_seed")]
        p[v] = p[seeds].mean(axis=1)
        p = p.rename(columns={c: f"{v}_{c}" for c in seeds})
        keep = ["answer_id", "span_id", "dataset", "label", v, *(f"{v}_{c}" for c in seeds)]
        df = p[keep] if df is None else df.merge(p[[k for k in keep if k not in
                                                    ("dataset", "label")]],
                                                 on=["answer_id", "span_id"])  # fmt: skip
    assert df is not None
    df = df.merge(spans, on=["answer_id", "span_id"], validate="one_to_one")
    probs = np.load(npz_path)
    rows = zip(df["answer_id"], df["start"], df["end"], strict=True)
    df["paper"] = [float(probs[a.replace("/", "__")][s : e + 1].max()) for a, s, e in rows]
    if lora_npz is not None:
        lora = np.load(lora_npz)
        rows = zip(df["answer_id"], df["start"], df["end"], strict=True)
        df["lora"] = [float(lora[a.replace("/", "__")][s : e + 1].max()) for a, s, e in rows]
    return df


def compare(df: pd.DataFrame, n_boot: int = 2000, model: str = "qwen25") -> dict:
    variants, reported = MODELS[model]["variants"], MODELS[model]["reported"]
    out = {}
    for ds, g in df.groupby("dataset"):
        y, groups = g["label"].to_numpy(), g["answer_id"].to_numpy()
        res = {"spans": len(g), "answers": int(g["answer_id"].nunique()),
               "paper_reported": reported[str(ds)],
               "paper_reported_lora": MODELS[model]["reported_lora"].get(str(ds))}  # fmt: skip
        cols = ["paper", *variants, *(["lora"] if "lora" in g else [])]
        for c in cols:
            s = g[c].to_numpy()
            res[c] = {
                "auroc": round(auroc(y, s), 4),
                "recall_at_10pct_fpr": round(recall_at_fpr(y, s), 4),
            }
        for v in variants:
            d = paired_bootstrap(y, g[v].to_numpy(), g["paper"].to_numpy(), groups, n_boot=n_boot)
            res[f"{v}_minus_paper"] = {k: round(x, 4) for k, x in d.items()}
        d = paired_bootstrap(y, g[variants[0]].to_numpy(), g[variants[1]].to_numpy(), groups,
                             n_boot=n_boot)  # fmt: skip
        res["layer_gain"] = {k: round(x, 4) for k, x in d.items()}  # chosen layer vs layer 26
        if "lora" in g:  # follow-ups B1: ours minus the released LoRA probe, same spans
            d = paired_bootstrap(y, g[variants[0]].to_numpy(), g["lora"].to_numpy(), groups,
                                 n_boot=n_boot)  # fmt: skip
            res[f"{variants[0]}_minus_lora"] = {k: round(x, 4) for k, x in d.items()}
        seeds = sorted(c for c in g if c.startswith(f"{variants[0]}_p_seed"))
        res[f"{variants[0]}_per_seed_auroc"] = [round(auroc(y, g[c].to_numpy()), 4) for c in seeds]
        out[str(ds)] = res
    out["claim_beats_paper"] = all(out[ds][f"{variants[0]}_minus_paper"]["ci_low"] > 0
                                   for ds in reported)  # fmt: skip
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--model=")]
    model = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--model=")), "qwen25")
    lora = Path(args[3]) if len(args) > 3 else None
    df = load(Path(args[0]), Path(args[1]), Path(args[2]), lora, model)
    res = compare(df, model=model)
    print(json.dumps(res, indent=1))
    name = "paper_h2h.json" if model == "qwen25" else f"paper_h2h_{model}.json"
    Path("results", name).write_text(json.dumps(res, indent=1))
