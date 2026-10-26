import json
import math
import re

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from hallustream import entities
from hallustream.detector import Detector, Entity, highlight
from hallustream.probe import forward

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
ANSWER = "Ada Lovelace was born in 1815 in London. She worked with Charles Babbage."


def words(text, sents):
    """Every word of the given sentences (a stand-in for GLiNER in tests)."""
    return [(lo + m.start(), lo + m.end()) for lo, hi in sents
            for m in re.finditer(r"\w+", text[lo:hi])]  # fmt: skip


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.manual_seed(0)
    lm = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    d, n = lm.config.hidden_size, lm.config.num_hidden_layers
    w = tmp_path_factory.mktemp("weights")
    for k in range(3):
        probe = {"weight": torch.randn(d), "bias": torch.zeros(1)}
        save_file(probe, str(w / f"probe_seed{k}.safetensors"))
    det = {"candidate": "B", "layer": n - 2, "offset": True, "b": -0.2,
           "w": (np.random.default_rng(0).normal(size=d) * 0.1).tolist(),
           "h_mean": [0.0] * d, "h_std": [1.0] * d}  # fmt: skip
    (w / "v3.json").write_text(json.dumps({"detector": det}))
    (w / "model.json").write_text(json.dumps({"hf_id": TINY, "layer": n - 2}))
    (w / "thresholds.json").write_text(json.dumps({"balanced": {"threshold": 0.5}}))
    return Detector.from_pretrained(w, model=lm, tokenizer=tok, entity_finder=words)


def test_score_is_the_probe_math(tiny):
    r = tiny.score("Tell me about Ada Lovelace.", ANSWER)
    assert [e.text for e in r.entities][:3] == ["Ada", "Lovelace", "was"]
    msgs = [{"role": "user", "content": "Tell me about Ada Lovelace."}]
    fw = forward(tiny.model, tiny.tokenizer, tiny.probe, msgs, ANSWER)
    lo = ANSWER.index("1815")
    toks = [i for i, (s, e) in enumerate(fw.offsets) if e > s >= 0 and s < lo + 4 and e > lo]
    _, v3 = tiny.probe.span_scores(fw, [(toks[0], toks[-1])])
    (e,) = [e for e in r.entities if e.text == "1815"]
    assert math.isclose(e.risk, 1 / (1 + math.exp(-v3[0])), rel_tol=1e-6)
    assert all(e.flagged == (e.risk >= 0.5) for e in r.entities) and r.threshold == 0.5


def test_thresholds_by_name_or_number(tiny):
    every = [(e.start, e.end) for e in tiny.score("q", ANSWER).entities]
    assert [(e.start, e.end) for e in tiny.score("q", ANSWER, threshold=0.0).flagged] == every
    assert tiny.score("q", ANSWER, threshold=1.01).flagged == []
    with pytest.raises(ValueError, match="unknown threshold"):
        tiny.score("q", ANSWER, threshold="strict")


def test_highlight_merges_overlaps_and_escapes_html():
    text = "Born in <Paris> in 1815."
    ents = [Entity("<Paris>", 8, 15, 0.9, True), Entity("Paris", 9, 14, 0.8, True),
            Entity("1815", 19, 23, 0.7, True)]  # fmt: skip
    assert highlight(text, ents) == "Born in [[<Paris>]] in [[1815]]."
    html = highlight(text, ents, "html")
    assert html.count("<mark") == 2 and "&lt;Paris&gt;" in html and 'title="risk 0.90"' in html
    assert "\033[" in highlight(text, ents, "ansi")


def test_stream_matches_score_on_the_same_text(tiny):
    events = list(tiny.stream("Tell me about Ada Lovelace.", max_new_tokens=60))
    text = "".join(e.text for e in events)
    assert events[-1].done and text
    streamed = {(e.start, e.end): e.risk for ev in events for e in ev.entities}
    assert streamed, "the stream scored no entities"
    # Teacher forcing re-tokenizes the text: compare entities that end before the first token
    # that differs from the generated one (all of them when nothing differs).
    msgs = [{"role": "user", "content": "Tell me about Ada Lovelace."}]
    fw = forward(tiny.model, tiny.tokenizer, tiny.probe, msgs, text)
    gen, again = events[-1].ids, fw.ids[fw.answer_start :]
    same = next((i for i, (a, b) in enumerate(zip(gen, again, strict=False)) if a != b), len(gen))
    cut = fw.offsets[fw.answer_start + same][0] if same < len(gen) else len(text)
    scored = tiny.score(msgs[0]["content"], text, spans=list(streamed))
    ref = {(e.start, e.end): e.risk for e in scored.entities}
    compared = [k for k in streamed if k[1] <= cut and k in ref]
    assert compared
    assert all(math.isclose(streamed[k], ref[k], abs_tol=1e-4) for k in compared)


def test_finding_entities_without_gliner_explains_what_to_do(tiny):
    if entities.available():
        pytest.skip("GLiNER is installed")
    plain = Detector(tiny.model, tiny.tokenizer, tiny.probe, 0.5)
    with pytest.raises(ImportError, match="hallustream\\[entities\\]"):
        plain.score("q", ANSWER)
