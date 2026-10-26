"""Stream an answer with the facts the model is probably making up highlighted.

    hallustream "Tell me about Lake Tahoe."

Each sentence is printed once it has been checked, with risky entities in red.
"""

from __future__ import annotations

import argparse
import json
import sys

from hallustream.detector import Detector, Entity, highlight


def _segment(text: str, lo: int, hi: int, flagged: list[Entity], color: bool) -> str:
    inside = [Entity(e.text, e.start - lo, e.end - lo, e.risk, True)
              for e in flagged if lo <= e.start and e.end <= hi]  # fmt: skip
    return highlight(text[lo:hi], inside, "ansi" if color else "plain")


def main(argv: list[str] | None = None, detector: Detector | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hallustream", description="Stream an answer with likely made-up facts highlighted."
    )
    p.add_argument("prompt")
    p.add_argument("--threshold", default="balanced", help="balanced, high_precision or a number")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--weights", default="qwen3-4b")
    p.add_argument("--device", default=None)
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--json", action="store_true", help="print every entity's risk at the end")
    a = p.parse_args(argv)
    try:
        threshold: str | float = float(a.threshold)
    except ValueError:
        threshold = a.threshold
    if detector is None:
        from transformers.utils import logging

        logging.disable_progress_bar()  # keep the terminal for the answer
        logging.set_verbosity_error()
    det = detector or Detector.from_pretrained(a.weights, device=a.device)
    color = not a.no_color and sys.stdout.isatty()
    text, printed, found = "", 0, []
    for ev in det.stream(a.prompt, max_new_tokens=a.max_new_tokens, threshold=threshold):
        text += ev.text
        found += ev.entities
        if ev.checked > printed:
            print(_segment(text, printed, ev.checked, [e for e in found if e.flagged], color),
                  end="", flush=True)  # fmt: skip
            printed = ev.checked
    flagged = [e for e in found if e.flagged]
    print(f"\n\n{len(found)} facts checked, {len(flagged)} flagged as likely made up.")
    if a.json:
        print(json.dumps([e.__dict__ for e in found], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
