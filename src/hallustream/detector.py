"""Flag the facts a model is probably making up, after it writes or while it writes.

    det = Detector.from_pretrained("qwen3-4b")
    report = det.score("Tell me about Lake Tahoe.", answer)   # text the model wrote
    print(report.highlight("ansi"))
    for event in det.stream("Tell me about Lake Tahoe."):    # flags as each sentence ends
        ...

The probe only knows what its own model knows: score text written by the model it was
trained for (Qwen3-4B-Instruct-2507 for the bundled weights), not other models' text.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch

from hallustream import entities as _entities
from hallustream.probe import Forward, Probe, char_to_tokens, forward
from hallustream.text import sentences

BUNDLED = ("qwen3-4b",)
SENTENCE_END = ".!?"
EntityFinder = Callable[[str, list[tuple[int, int]]], list[tuple[int, int]]]


@dataclass
class Entity:
    text: str
    start: int  # character offsets in the answer
    end: int
    risk: float  # probability that the entity is made up (sigmoid of the v3 score)
    flagged: bool


@dataclass
class Report:
    text: str
    entities: list[Entity]
    threshold: float

    @property
    def flagged(self) -> list[Entity]:
        return [e for e in self.entities if e.flagged]

    def highlight(self, style: str = "plain") -> str:
        """The text with flagged entities marked: "plain" [[...]], "ansi" colors, or "html"."""
        return highlight(self.text, self.flagged, style)


@dataclass
class Event:
    """One step of stream(): new text, and the entities of sentences that just finished."""

    text: str
    entities: list[Entity] = field(default_factory=list)
    done: bool = False
    ids: list[int] = field(default_factory=list)  # on the last event: every generated token


def highlight(text: str, flagged: list[Entity], style: str = "plain") -> str:
    spans: list[list] = []
    for e in sorted(flagged, key=lambda e: e.start):
        if spans and e.start < spans[-1][1]:
            spans[-1][1], spans[-1][2] = max(spans[-1][1], e.end), max(spans[-1][2], e.risk)
        else:
            spans.append([e.start, e.end, e.risk])
    esc = html.escape if style == "html" else (lambda s: s)
    out, pos = [], 0
    for lo, hi, risk in spans:
        out.append(esc(text[pos:lo]))
        inner = esc(text[lo:hi])
        if style == "html":
            out.append(f'<mark class="hs-risk" title="risk {risk:.2f}">{inner}</mark>')
        elif style == "ansi":
            out.append(f"\033[1;97;41m{inner}\033[0m")
        else:
            out.append(f"[[{inner}]]")
        pos = hi
    out.append(esc(text[pos:]))
    return "".join(out)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _weights_dir(weights: str | Path) -> Path:
    if str(weights) in BUNDLED:
        return Path(str(files("hallustream") / "weights" / str(weights)))
    return Path(weights)


class Detector:
    """A model, its probe, and the thresholds for flagging an entity."""

    def __init__(self, model, tokenizer, probe: Probe, threshold: str | float = 0.5,
                 chat_template_kwargs: dict | None = None, thresholds: dict | None = None,
                 entity_finder: EntityFinder | None = None):  # fmt: skip
        self.model, self.tokenizer, self.probe = model, tokenizer, probe
        self.chat = chat_template_kwargs or {}
        self.thresholds = thresholds or {}
        self.threshold = self._threshold(threshold)
        self.find_entities = entity_finder or _entities.find

    @classmethod
    def from_pretrained(cls, weights: str | Path = "qwen3-4b", model=None, tokenizer=None,
                        device: str | None = None, dtype: torch.dtype = torch.bfloat16,
                        threshold: str | float = "balanced",
                        entity_finder: EntityFinder | None = None) -> Detector:  # fmt: skip
        """Load bundled weights by name ("qwen3-4b") or a weights folder, and the model they
        were trained for (unless model and tokenizer are given)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        d = _weights_dir(weights)
        meta = json.loads((d / "model.json").read_text())
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(meta["hf_id"], dtype=dtype)
            model.to(device or _default_device())  # pyright: ignore[reportArgumentType]
            model.eval()
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(meta["hf_id"])
        probe = Probe(sorted(d.glob("probe_seed*.safetensors")), d / "v3.json",
                      device=str(model.device))  # fmt: skip
        if probe.weight.shape[0] != model.config.hidden_size:
            raise ValueError(f"probe width {probe.weight.shape[0]} does not match the model")
        th = d / "thresholds.json"
        return cls(model, tokenizer, probe, threshold, meta.get("chat_template_kwargs"),
                   json.loads(th.read_text()) if th.exists() else {}, entity_finder)  # fmt: skip

    def _threshold(self, threshold: str | float | None) -> float:
        if threshold is None:
            return self.threshold
        if isinstance(threshold, str):
            if threshold not in self.thresholds:
                raise ValueError(f"unknown threshold {threshold!r}; have {list(self.thresholds)}")
            return float(self.thresholds[threshold]["threshold"])
        return float(threshold)

    def _messages(self, prompt: str | None, messages: list[dict] | None) -> list[dict]:
        if messages is None:
            if prompt is None:
                raise ValueError("give a prompt or messages")
            messages = [{"role": "user", "content": prompt}]
        return messages

    def _entities(self, text: str, fw: Forward, spans, threshold: float) -> list[Entity]:
        keep, tok_spans = [], []
        for lo, hi in sorted(set(spans)):
            if (t := char_to_tokens(fw, lo, hi)) is not None:
                keep.append((lo, hi))
                tok_spans.append(t)
        _, v3 = self.probe.span_scores(fw, tok_spans)
        risk = [1 / (1 + math.exp(-float(z))) for z in v3]
        return [Entity(text[lo:hi], lo, hi, r, r >= threshold)
                for (lo, hi), r in zip(keep, risk, strict=True)]  # fmt: skip

    def score(self, prompt: str | None = None, text: str = "", *, messages: list[dict] | None
              = None, spans: list[tuple[int, int]] | None = None,
              threshold: str | float | None = None) -> Report:  # fmt: skip
        """Risk of every entity in `text`, an answer the model wrote to `prompt` (or after
        `messages`). Entities are found automatically unless `spans` are given."""
        th = self._threshold(threshold)
        fw = forward(self.model, self.tokenizer, self.probe, self._messages(prompt, messages),
                     text, **self.chat)  # fmt: skip
        if spans is None:
            spans = self.find_entities(text, sentences(text))
        return Report(text, self._entities(text, fw, spans, th), th)

    @torch.no_grad()
    def stream(self, prompt: str | None = None, *, messages: list[dict] | None = None,
               max_new_tokens: int = 512, threshold: str | float | None = None,
               temperature: float = 0.0) -> Iterator[Event]:  # fmt: skip
        """Generate an answer token by token; when a sentence ends, its entities are scored
        from the hidden states already computed for generation (no second pass)."""
        from transformers import DynamicCache

        th = self._threshold(threshold)
        tok, lm = self.tokenizer, self.model
        msgs = self._messages(prompt, messages)
        prefix = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, **self.chat
        )
        ids = tok(prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(lm.device)
        eos = lm.generation_config.eos_token_id
        stop = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}
        cache = DynamicCache()
        out = lm(ids, past_key_values=cache, use_cache=True)
        gen: list[int] = []
        hidden: list[np.ndarray] = []
        probs: list[np.ndarray] = []
        text, scored, seen = "", set(), set()  # sentences and entity spans already scored
        for _ in range(max_new_tokens):
            logits = out.logits[0, -1].float()
            if temperature > 0:
                nxt = int(torch.multinomial(torch.softmax(logits / temperature, -1), 1))
            else:
                nxt = int(logits.argmax())
            if nxt in stop:
                break
            gen.append(nxt)
            out = lm(torch.tensor([[nxt]], device=lm.device), past_key_values=cache,
                     use_cache=True, output_hidden_states=True)  # fmt: skip
            h = out.hidden_states[self.probe.layer + 1][0, -1]
            hidden.append(h.half().cpu().numpy())
            probs.append(self.probe.token_probs(h[None])[0])
            ended = text.rstrip()[-1:] in SENTENCE_END or "\n" in text[-1:]
            new = tok.decode(gen, skip_special_tokens=True)
            delta = new[len(text) :] if new.startswith(text) else new
            text = new
            found = []
            if ended:
                found = self._finished(text, gen, hidden, probs, scored, seen, th, False)
            yield Event(delta, found)
        last = self._finished(text, gen, hidden, probs, scored, seen, th, True)
        yield Event("", last, True, gen)

    def _finished(self, text, gen, hidden, probs, scored, seen, th, final) -> list[Entity]:
        """Entities of sentences that are complete and not scored yet (each span once, even if
        sentence boundaries shift as more text arrives)."""
        sents = sentences(text)
        todo = [s for s in (sents if final else sents[:-1]) if s not in scored]
        if not todo or not gen:
            return []
        scored.update(todo)
        dec = [len(self.tokenizer.decode(gen[:i], skip_special_tokens=True))
               for i in range(len(gen) + 1)]  # fmt: skip
        n = len(gen)
        fw = Forward(gen, list(zip(dec[:-1], dec[1:], strict=True)), 0, np.stack(probs),
                     np.stack(hidden), np.full(n, np.nan), np.full(n, np.nan))  # fmt: skip
        spans = [s for s in self.find_entities(text, todo) if s not in seen]
        seen.update(spans)
        return self._entities(text, fw, spans, th)
