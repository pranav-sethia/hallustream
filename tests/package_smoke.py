"""Smoke test of the installed package, as a new user gets it (run by CI outside the repo).

Installs nothing itself: run it with the Python of a fresh environment that has only the built
wheel with [entities]. It builds tiny probe weights for a tiny model, then uses the real GLiNER
entity finder through the Python API and the command line.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import hallustream
from hallustream import cli
from hallustream.detector import Detector

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
ANSWER = "Ada Lovelace was born in 1815 in London. She worked with Charles Babbage."

assert "site-packages" in hallustream.__file__, f"not the installed package: {hallustream.__file__}"
torch.manual_seed(0)
lm = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(TINY)
d, n = lm.config.hidden_size, lm.config.num_hidden_layers
w = Path(tempfile.mkdtemp())
for k in range(2):
    save_file(
        {"weight": torch.randn(d), "bias": torch.zeros(1)}, str(w / f"probe_seed{k}.safetensors")
    )
det = {"candidate": "B", "layer": n - 2, "offset": True, "b": 0.0, "w": [0.0] * d,
       "h_mean": [0.0] * d, "h_std": [1.0] * d}  # fmt: skip
(w / "detector.json").write_text(json.dumps({"detector": det}))
(w / "model.json").write_text(json.dumps({"hf_id": TINY}))
(w / "thresholds.json").write_text(json.dumps({"balanced": {"threshold": 0.5}}))

detector = Detector.from_pretrained(w, model=lm, tokenizer=tok)  # default finder: GLiNER
report = detector.score("Tell me about Ada Lovelace.", ANSWER)
found = {e.text for e in report.entities}
assert {"Ada Lovelace", "1815", "London"} <= found, found
assert all(0 < e.risk < 1 for e in report.entities)
events = list(detector.stream("Hi.", max_new_tokens=8))
assert events[-1].done and np.isfinite([e.risk for ev in events for e in ev.entities]).all()

answer = w / "answer.txt"
answer.write_text(ANSWER)
assert cli.main(["score", "Tell me about Ada Lovelace.", str(answer), "--json"], detector) == 0
print("ok:", sorted(found), file=sys.stderr)
