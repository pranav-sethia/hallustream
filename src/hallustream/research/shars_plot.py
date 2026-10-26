"""Chart for the SHARS test: false claims per biography vs detection time per fact.

Run: uv run python -m hallustream.research.shars_plot <report.json> <out.png>
(report.json from hallustream.research.shars_test on results/shars/test60)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = {  # arm -> (label, color)
    "halluse": ("SHARS + its own detector (HalluSE)", "#8a8a8a"),
    "logprob_45": ("SHARS + token surprise", "#c4a35a"),
    "probe_45": ("SHARS + probe", "#2a7de1"),
    "probe_60": ("SHARS + probe, stricter", "#14508f"),
}


def plot(report: dict, out: Path) -> None:
    arms = report["arms"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    greedy = arms["naive"]["unsupported_per_bio"]
    ax.axhline(greedy, color="#bbbbbb", ls="--", lw=1)
    ax.text(0.02, greedy + 0.12, f"plain writing: {greedy:.1f} false claims", color="#777777",
            fontsize=8.5)  # fmt: skip
    for key, (label, color) in ARMS.items():
        a = arms[key]
        x, y = a["detect_seconds_per_prop"], a["unsupported_per_bio"]
        ax.scatter(x, y, s=90, color=color, zorder=3)
        ax.annotate(f"{label}\n{y:.1f} false, {a['supported_per_bio']:.1f} true, "
                    f"precision {a['precision']:.0%}", (x, y), xytext=(8, -4),
                    textcoords="offset points", fontsize=8, color=color, va="center")  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(0.02, 60)
    ax.set_ylim(0, greedy + 1)
    ax.set_xlabel("Detection time per fact (seconds, log scale, one L4 GPU)")
    ax.set_ylabel("False claims per biography")
    ax.set_title("SHARS on Qwen3-4B, 60 held-out FActScore people", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


if __name__ == "__main__":
    plot(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
