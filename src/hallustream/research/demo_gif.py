"""Render a recorded stream() run (modal_jobs/library_check.py showcase) as a terminal GIF.

Text appears gray as the model writes it, turns white once its sentence is checked, and
flagged entities get a red background. Timing follows the recording.

Run: uv run python -m hallustream.research.demo_gif <showcase.json> <prompt substring> <out.gif>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from hallustream.detector import Entity, Report

FONT = "/System/Library/Fonts/Menlo.ttc"
W, COLS, SIZE, PAD, LINE = 980, 82, 17, 22, 26
BG, GRAY, WHITE, RED, DIM, GREEN = "#14161a", "#6f7680", "#e8eaed", "#c62828", "#9aa0a8", "#7fd1a0"
FPS = 12


def _layout(text: str) -> list[list[tuple[int, str]]]:
    """Wrap text into lines of (char index, char), breaking at spaces and newlines."""
    lines, line, col = [], [], 0
    words: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    for i, ch in enumerate(text):
        if ch in " \n":
            if cur:
                words.append(cur)
                cur = []
            words.append([(i, ch)])
        else:
            cur.append((i, ch))
    if cur:
        words.append(cur)
    for w in words:
        if w[0][1] == "\n":
            lines.append(line)
            line, col = [], 0
            continue
        if col + len(w) > COLS and w[0][1] != " ":
            lines.append(line)
            line, col = [], 0
        if not line and w[0][1] == " ":
            continue
        line += w
        col += len(w)
    lines.append(line)
    return lines


def render(run: dict, out: Path, hold: float = 3.5) -> None:
    font = ImageFont.truetype(FONT, SIZE)
    bold = ImageFont.truetype(FONT, SIZE, index=1)
    events = run["events"]
    text = "".join(e["text"] for e in events)
    lines = _layout(text)
    height = PAD * 2 + LINE * (len(lines) + 5)
    frames, durations = [], []
    t_end = events[-1]["t"]
    n_frames = int(t_end * FPS) + 1
    found = [Entity(**x) for e in events for x in e["entities"]]
    summary = Report(text, found, 0.0).summary().splitlines()[0].rstrip(":") + " (red)"
    for f in range(n_frames + 1):
        t = min(f / FPS, t_end)
        shown = [e for e in events if e["t"] <= t]
        n_chars = sum(len(e["text"]) for e in shown)
        checked = max((e["checked"] for e in shown), default=0)
        flags = [(x["start"], x["end"]) for e in shown for x in e["entities"] if x["flagged"]]
        img = Image.new("RGB", (W, height), BG)
        d = ImageDraw.Draw(img)
        d.text((PAD, PAD), "$ ", font=font, fill=GREEN)
        d.text((PAD + font.getlength("$ "), PAD), f'hallustream "{run["prompt"]}"', font=bold,
               fill=WHITE)  # fmt: skip
        y = PAD + LINE * 2
        for line in lines:
            x = PAD
            for i, ch in line:
                if i >= n_chars:
                    break
                cw = font.getlength(ch)
                red = i < checked and any(a <= i < b for a, b in flags)
                if red:
                    d.rectangle([x, y - 2, x + cw, y + LINE - 6], fill=RED)
                d.text((x, y), ch, font=bold if red else font,
                       fill=WHITE if (red or i < checked) else GRAY)  # fmt: skip
                x += cw
            y += LINE
        if f == n_frames:
            d.text((PAD, y + LINE), summary, font=font, fill=DIM)
        frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE, colors=32))
        durations.append(int(hold * 1000) if f == n_frames else int(1000 / FPS))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=True)  # fmt: skip


if __name__ == "__main__":
    runs = json.loads(Path(sys.argv[1]).read_text())
    run = next(r for r in runs if sys.argv[2] in r["prompt"])
    out = Path(sys.argv[3])
    render(run, out)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
