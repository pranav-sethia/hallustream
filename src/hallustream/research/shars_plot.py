"""Chart for the SHARS test: false claims per biography vs detection time per fact.

Run: uv run python -m hallustream.research.shars_plot <report.json> <out.png>
(report.json from hallustream.research.shars_test on results/shars/test182)

     uv run python -m hallustream.research.shars_plot tradeoff <followup.json> <out.png>
(followup.json from hallustream.research.shars_followup: false vs true claims per person for
every setting of each detector, EVAL_PROTOCOL.md follow-ups A)
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
    n = arms["naive"]["bios"]
    ax.set_title(f"SHARS on Qwen3-4B, {n} held-out FActScore people", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


# Each detector's settings, from fewest to most flags; colors validated with the dataviz checks
CURVES = {
    "HalluStream": (["probe_30", "probe_45", "probe_60"], "#2a78d6"),
    "SHARS's detector (HalluSE)": (["halluse", "halluse_07"], "#eb6834"),
    "Token surprise": (["logprob_45"], "#1baf7a"),
}
SINGLES = {"naive": "Qwen3-4B alone", "probe_v1": "Probe trained the paper's way"}


def tradeoff(res: dict, out: Path, labels: str = "shared") -> None:
    arms = res[labels]["arms"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    ink, muted = "#0b0b0b", "#52514e"
    for name, (keys, color) in CURVES.items():
        pts = [(arms[k]["supported_per_bio"], arms[k]["unsupported_per_bio"], k)
               for k in keys if k in arms]  # fmt: skip
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=color, lw=2, zorder=2)
        ax.scatter(xs, ys, s=64, color=color, edgecolor="#fcfcfb", linewidth=2, zorder=3,
                   label=name)  # fmt: skip
        for x, y, k in pts:
            a = arms[k]
            ax.annotate(f"{a['response_rate']:.0%} answered", (x, y), xytext=(7, -12),
                        textcoords="offset points", fontsize=7.5, color=muted)  # fmt: skip
    for k, label in SINGLES.items():
        if k in arms:
            x, y = arms[k]["supported_per_bio"], arms[k]["unsupported_per_bio"]
            ax.scatter(x, y, s=56, color="#9a9994", edgecolor="#fcfcfb", linewidth=2, zorder=3)
            ax.annotate(label, (x, y), xytext=(7, -10), textcoords="offset points",
                        fontsize=8, color=ink)  # fmt: skip
    ax.set_xlabel("True claims per person (more is better)", color=ink)
    ax.set_ylabel("False claims per person (fewer is better)", color=ink)
    n = arms["naive"]["bios"]
    ax.set_title(f"SHARS on Qwen3-4B: {n} held-out FActScore people, each detector at "
                 "several settings", fontsize=9.5, color=ink)  # fmt: skip
    ax.grid(color="#e8e7e3", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="#fcfcfb")


if __name__ == "__main__":
    if sys.argv[1] == "tradeoff":
        tradeoff(json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3]))
    else:
        plot(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
