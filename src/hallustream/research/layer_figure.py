"""Chart for RESULTS.md: validation AUROC of a probe on each layer, by relative depth, for the
two models used (Qwen3-4B, every layer; Qwen2.5-7B, the head-to-head sweep). Both use the same
L2 logistic regression (C = 3e-5). The paper's default layer, 0.95 of depth, is marked.

Run: uv run python -m hallustream.research.layer_figure results docs/img/layer_curve.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = [  # (sweep file, label, number of layers, color)
    ("full_qwen3_fair.json", "Qwen3-4B (every layer)", 36, "#0a6c99"),
    ("main_qwen25_singlesC.json", "Qwen2.5-7B (head-to-head with the paper)", 28, "#c4a35a"),
]


def plot(root: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    for name, label, n, color in MODELS:
        cands = json.loads((root / name).read_text())["candidates"]
        pts = sorted((c["layers"][0], c["val_auroc"]) for c in cands if len(c["layers"]) == 1)
        xs, ys = [(layer + 1) / n for layer, _ in pts], [v for _, v in pts]
        ax.plot(xs, ys, color=color, lw=2, marker="o" if len(pts) < 20 else None, ms=4,
                label=label)  # fmt: skip
        best = max(pts, key=lambda p: p[1])
        ax.annotate(f"layer {best[0]}: {best[1]:.3f}", ((best[0] + 1) / n, best[1]),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8,
                    color=color)  # fmt: skip
    ax.axvline(0.95, color="#999999", ls="--", lw=1)
    ax.text(0.94, 0.705, "paper's default\nlayer (0.95 of depth)", ha="right", fontsize=8,
            color="#777777")  # fmt: skip
    ax.set_xlabel("Layer, as a share of the model's depth")
    ax.set_ylabel("Validation AUROC")
    ax.set_xlim(0, 1)
    ax.set_title("A probe on each layer: the signal peaks in the middle", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    plot(Path(sys.argv[1]), Path(sys.argv[2]))
