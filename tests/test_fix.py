import json

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from hallustream.fix import score, spans

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
BIO = (
    "**Ada Lovelace** (1815–1852) was a mathematician. She was born in London.\n\n"
    "### Work\n- Wrote notes in 1843."
)


def test_sentences_skip_headers_and_split_lines():
    got = [BIO[a:b] for a, b in spans.sentences(BIO)]
    assert got == [
        "**Ada Lovelace** (1815–1852) was a mathematician.",
        "She was born in London.",
        "- Wrote notes in 1843.",
    ]


def test_split_people_is_fixed():
    names = [f"p{i}" for i in range(500)]
    dev, test = spans.split_people(names)
    assert len(dev) == 100 and len(test) == 400 and not set(dev) & set(test)
    assert spans.split_people(list(reversed(names))) == (dev, test)


def _detector(tmp_path, d):
    files = []
    for k in range(2):
        f = tmp_path / f"seed{k}.safetensors"
        save_file({"weight": torch.randn(d), "bias": torch.zeros(())}, str(f))
        files.append(f)
    det = {"candidate": "B", "layer": score.LAYER, "offset": True, "b": 0.5, "w": [0.0] * d,
           "h_mean": [0.0] * d, "h_std": [1.0] * d}  # fmt: skip
    (tmp_path / "v3.json").write_text(json.dumps({"detector": det}))
    return score.Detector(files, tmp_path / "v3.json", device="cpu")


def test_forward_and_signals_line_up(tmp_path, monkeypatch):
    tok = AutoTokenizer.from_pretrained(TINY)
    lm = AutoModelForCausalLM.from_pretrained(TINY).eval()
    monkeypatch.setattr(score, "LAYER", lm.config.num_hidden_layers - 2)
    det = _detector(tmp_path, lm.config.hidden_size)
    msgs = [{"role": "user", "content": "Tell me a bio of Ada Lovelace."}]
    fw = score.forward(lm, tok, det, msgs, BIO)
    assert fw.probs.shape == (len(fw.ids), 2) and fw.hidden.shape[0] == len(fw.ids)
    lo = BIO.index("1815")
    a, b = score.char_to_tokens(fw, lo, lo + 4)
    assert tok.decode(fw.ids[a : b + 1]).strip() == "1815"
    v1, v3 = det.span_scores(fw, [(a, b)])
    p = fw.probs[a : b + 1].max(0).mean()
    assert np.isclose(v1[0], np.log(p / (1 - p)), atol=1e-4) and np.isclose(v3[0], v1[0] + 0.5)
    sents = spans.sentences(BIO)
    sig = score.sentence_signals(fw, det, sents[0], [(lo, lo + 4)])
    assert sig["n_entities"] == 1 and np.isclose(sig["v3_max"], v3[0])
