"""Summarize sweep results: rank candidates, test each against the best, plot AUROC by depth.

Each candidate is compared with the best one by a paired bootstrap over validation
questions. A candidate whose 95% interval for (best - candidate) includes 0 is
reported as tied with the best.

Run: uv run python -m hallustream.research.sweep_report results/full_qwen35_singles.json ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from hallustream.research.data import load_records, question_key
from hallustream.research.metrics import paired_bootstrap

QWEN35_FULL_ATTENTION = set(range(3, 32, 4))  # 0-based decoder layers; the rest are DeltaNet


def name(c: dict) -> str:
    layers = "+".join(map(str, c["layers"]))
    out = f"mean({layers})" if c["combine"] == "mean" and len(c["layers"]) > 1 else layers
    return out if c.get("C", 1.0) == 1.0 else f"{out} C={c['C']}"


def compare_to_best(files: list[Path], q_of: dict[str, str], n_boot: int = 1000) -> dict:
    cands = [c for f in files for c in json.loads(f.read_text())["candidates"]]
    first = cands[0]
    for c in cands:  # every candidate must be scored on the same spans
        assert c["span_answer_ids"] == first["span_answer_ids"]
    y = np.array(first["labels"])
    groups = np.array([q_of[a] for a in first["span_answer_ids"]])
    best = max(cands, key=lambda c: c["val_auroc"])
    b = np.array(best["scores"])
    table = []
    for c in sorted(cands, key=lambda c: -c["val_auroc"]):
        row = {"candidate": name(c), "layers": c["layers"], "combine": c["combine"]}
        row |= {"val_auroc": c["val_auroc"], "recall_at_10pct_fpr": c["val_recall_at_10pct_fpr"]}
        if c is not best:
            bs = paired_bootstrap(y, b, np.array(c["scores"]), groups, n_boot=n_boot)
            row |= {"gap_to_best": round(bs["delta"], 4)}
            row |= {"gap_ci": [round(bs["ci_low"], 4), round(bs["ci_high"], 4)]}
            row |= {"tied_with_best": bs["ci_low"] <= 0}
        table.append(row)
    base = json.loads(files[0].read_text())["baselines"]
    return {"best": name(best), "n_val_spans": len(y), "baselines": base, "table": table}


def plot(singles: dict[str, dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, rep in singles.items():
        rows = sorted((r for r in rep["table"] if len(r["layers"]) == 1), key=lambda r: r["layers"])
        layers = [r["layers"][0] for r in rows]
        n = max(layers) + 1
        depth = [(layer + 1) / n for layer in layers]
        ax.plot(depth, [r["val_auroc"] for r in rows], marker="o", ms=3, label=label)
        if "3.5" in label:
            full = [
                (d, r["val_auroc"])
                for d, r in zip(depth, rows, strict=True)
                if r["layers"][0] in QWEN35_FULL_ATTENTION
            ]
            ax.scatter(
                *zip(*full, strict=True),
                s=40,
                facecolors="none",
                edgecolors="k",
                zorder=3,
                label="Qwen3.5 full-attention layers",
            )
    best_base = max(max(rep["baselines"].values()) for rep in singles.values())
    ax.axhline(best_base, ls="--", c="grey", lw=1, label="best confidence baseline")
    ax.set_xlabel("depth through the model (layer / total layers)")
    ax.set_ylabel("validation AUROC (entity level)")
    ax.set_title("Where does the hallucination signal live?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)


if __name__ == "__main__":
    q_of = {r.answer_id: question_key(r.question) for r in load_records()}
    reports = {}
    for arg in sys.argv[1:]:  # label=file1,file2
        label, files = arg.split("=")
        paths = [Path(f) for f in files.split(",")]
        reports[label] = compare_to_best(paths, q_of)
        print(label, "best:", reports[label]["best"])
        for r in reports[label]["table"][:12]:
            print("  ", {k: v for k, v in r.items() if k not in ("layers", "combine")})
    Path("results/sweep_report.json").write_text(json.dumps(reports, indent=1))
    plot(reports, Path("results/layer_curve.png"))
