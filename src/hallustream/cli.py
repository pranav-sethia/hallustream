"""The hallustream command: answers with the details the model is probably making up highlighted.

    hallustream "Tell me about Lake Tahoe."               ask one question (the default)
    hallustream chat                                      have a conversation
    hallustream score "Tell me about Lake Tahoe." a.txt   check an answer it already wrote
    hallustream batch prompts.jsonl -o results.jsonl      check many prompts or answers

Answers appear sentence by sentence, once each sentence has been checked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hallustream import __version__
from hallustream.detector import BUNDLED, Detector, Entity, Report, _weights_dir, highlight

COMMANDS = ("ask", "chat", "score", "batch")
DIM, RESET = "\033[2m", "\033[0m"


def _parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--threshold", default="balanced",
                        help="balanced (default), strict, or a risk between 0 and 1")  # fmt: skip
    shared.add_argument("--weights", default="qwen3-4b", help="bundled name or weights folder")
    shared.add_argument("--device", choices=["cuda", "mps", "cpu"],
                        help="default: the fastest available")  # fmt: skip
    p = argparse.ArgumentParser(
        prog="hallustream",
        description="Answers from Qwen3-4B with the details it is probably making up highlighted."
        " Without a command, the arguments go to 'ask'.",
        epilog='Example: hallustream "Tell me about Lake Tahoe."  (see "hallustream ask -h" for'
        " --threshold, --device, --json and the other options)",
    )
    p.add_argument("--version", action="version", version=f"hallustream {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", parents=[shared], help="answer one question (the default)")
    ask.add_argument("prompt")
    chat = sub.add_parser("chat", parents=[shared], help="have a conversation")
    score = sub.add_parser("score", parents=[shared], help="check an answer it already wrote")
    score.add_argument("prompt", help="the question the answer was written for")
    score.add_argument("file", nargs="?", default="-", help="the answer (default: stdin)")
    batch = sub.add_parser("batch", parents=[shared], help="check a file of prompts or answers")
    batch.add_argument("input", help='JSONL with "prompt" (and "answer" to check one already '
                       'written), or one prompt per line; - for stdin')  # fmt: skip
    batch.add_argument("-o", "--output", default="-", help="JSONL results (default: stdout)")
    for s in (ask, chat, batch):
        s.add_argument("--max-new-tokens", type=int, default=512)
    for s in (ask, chat, score):
        s.add_argument("--no-color", action="store_true")
    for s in (ask, score):
        s.add_argument("--json", action="store_true", help="print one JSON object instead")
    return p


def _load(weights: str, device: str | None) -> Detector:
    import logging as pylogging
    import warnings

    import torch
    from huggingface_hub import try_to_load_from_cache
    from transformers.utils import logging

    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("hallustream: --device cuda, but no CUDA GPU is available")
    if device == "mps" and not torch.backends.mps.is_available():
        sys.exit("hallustream: --device mps, but this machine has no Apple GPU")
    # This is a command for reading answers: library warnings (GLiNER, transformers, torch's
    # macOS note about redirects) are turned off for the whole process.
    warnings.simplefilter("ignore")
    pylogging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel("ERROR")

    hf_id = json.loads((_weights_dir(weights) / "model.json").read_text())["hf_id"]
    if isinstance(try_to_load_from_cache(hf_id, "model.safetensors.index.json"), str):
        logging.disable_progress_bar()  # already downloaded: keep the terminal for the answer
    else:
        print(f"Downloading {hf_id} (about 8 GB, once)...", file=sys.stderr)
    logging.set_verbosity_error()
    return Detector.from_pretrained(weights, device=device)


def _segment(text: str, lo: int, hi: int, flagged: list[Entity], color: bool) -> str:
    inside = [Entity(e.text, e.start - lo, e.end - lo, e.risk, True)
              for e in flagged if lo <= e.start and e.end <= hi]  # fmt: skip
    return highlight(text[lo:hi], inside, "ansi" if color else "plain")


def _stream(det: Detector, messages: list[dict], max_new_tokens: int, threshold, color: bool,
            ) -> Report:  # fmt: skip
    """Print each sentence once it has been checked; return the whole answer's report."""
    text, printed, found, n = "", 0, [], 0
    for ev in det.stream(messages=messages, max_new_tokens=max_new_tokens, threshold=threshold):
        text += ev.text
        found += ev.entities
        n = len(ev.ids)
        if ev.checked > printed:
            print(_segment(text, printed, ev.checked, [e for e in found if e.flagged], color),
                  end="", flush=True)  # fmt: skip
            printed = ev.checked
    cut = f"\n(cut off at {max_new_tokens} tokens; raise --max-new-tokens for more)"
    print(_dim(cut, color) if n >= max_new_tokens else "")
    return Report(text, found, det._threshold(threshold))


def _dim(text: str, color: bool) -> str:
    return f"{DIM}{text}{RESET}" if color else text


def _legend(threshold: float, color: bool) -> str:
    return _dim(f"Risk is the estimated chance a detail is made up; flagged from {threshold:.0%}.",
                color)  # fmt: skip


def _ask(det: Detector, a, threshold, color: bool) -> int:
    if a.json:
        report = det.generate(a.prompt, max_new_tokens=a.max_new_tokens, threshold=threshold)
        print(json.dumps({"prompt": a.prompt, **report.to_dict()}, ensure_ascii=False))
        return 0
    msgs = [{"role": "user", "content": a.prompt}]
    report = _stream(det, msgs, a.max_new_tokens, threshold, color)
    print(f"\n{report.summary()}\n{_legend(report.threshold, color)}")
    return 0


def _chat(det: Detector, a, threshold, color: bool) -> int:
    legend = _legend(det._threshold(threshold), color)
    print(f"Chat with the model; likely made-up details are highlighted.\n{legend}\n"
          "Type exit or press Ctrl-D to quit.")  # fmt: skip
    messages: list[dict] = []
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.lower() in ("exit", "quit"):
            return 0
        if not line:
            continue
        messages.append({"role": "user", "content": line})
        report = _stream(det, messages, a.max_new_tokens, threshold, color)
        messages.append({"role": "assistant", "content": report.text})
        print(f"\n{report.summary()}")


def _score(det: Detector, a, threshold, color: bool) -> int:
    text = (sys.stdin.read() if a.file == "-" else Path(a.file).read_text()).rstrip()
    report = det.score(a.prompt, text, threshold=threshold)
    if a.json:
        print(json.dumps({"prompt": a.prompt, **report.to_dict()}, ensure_ascii=False))
        return 0
    print(report.highlight("ansi" if color else "plain"))
    print(f"\n{report.summary()}\n{_legend(report.threshold, color)}")
    return 0


def _batch(det: Detector, a, threshold, color: bool) -> int:
    raw = sys.stdin.read() if a.input == "-" else Path(a.input).read_text()
    lines = [x for x in raw.splitlines() if x.strip()]
    out = sys.stdout if a.output == "-" else open(a.output, "w")
    for i, line in enumerate(lines, 1):
        try:
            item: dict = (json.loads(line) if line.lstrip().startswith("{")
                          else {"prompt": line.strip()})  # fmt: skip
        except json.JSONDecodeError as e:
            sys.exit(f"{a.input} line {i}: not valid JSON ({e.msg})")
        prompt, msgs = item.get("prompt"), item.get("messages")
        if prompt is None and msgs is None:
            sys.exit(f'{a.input} line {i}: needs a "prompt" or "messages"')
        if "answer" in item:
            report = det.score(prompt, item["answer"], messages=msgs, threshold=threshold)
        else:
            report = det.generate(prompt, messages=msgs, max_new_tokens=a.max_new_tokens,
                                  threshold=threshold)  # fmt: skip
        out.write(json.dumps({**item, **report.to_dict()}, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{i}/{len(lines)}: {len(report.flagged)} flagged", file=sys.stderr)
    if out is not sys.stdout:
        out.close()
    return 0


def main(argv: list[str] | None = None, detector: Detector | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in (*COMMANDS, "-h", "--help", "--version"):
        argv.insert(0, "ask")
    p = _parser()
    a = p.parse_args(argv)
    if detector is None and not (_weights_dir(a.weights) / "model.json").exists():
        p.error(f"--weights {a.weights}: not a bundled name {list(BUNDLED)} or a weights folder")
    try:
        threshold: str | float = float(a.threshold)
        if not 0 <= threshold <= 1:
            p.error(f"--threshold {a.threshold}: a risk must be between 0 and 1")
    except ValueError:
        threshold = a.threshold
        file = _weights_dir(a.weights) / "thresholds.json"
        presets = detector.thresholds if detector else json.loads(file.read_text())
        if not isinstance(presets.get(threshold), dict):
            names = [k for k, v in presets.items() if isinstance(v, dict)]
            p.error(f"unknown threshold {threshold!r}; use one of {names} or a number")
    path = getattr(a, "file", None) or getattr(a, "input", None)
    if path and path != "-" and not Path(path).is_file():
        p.error(f"{path}: no such file")
    color = not getattr(a, "no_color", True) and sys.stdout.isatty()
    run = {"ask": _ask, "chat": _chat, "score": _score, "batch": _batch}[a.command]
    try:
        det = detector or _load(a.weights, a.device)
        return run(det, a, threshold, color)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
