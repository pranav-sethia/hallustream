# Using HalluStream

HalluStream flags the entities (names, dates, places, numbers, titles) that Qwen3-4B-Instruct-2507 is probably making up, from its own hidden states. It adds no extra generation and needs no search.

## Install
```bash
pip install "hallustream[entities] @ git+https://github.com/pranav-sethia/hallustream"
```
- `[entities]` adds [GLiNER](https://github.com/urchade/GLiNER) to find entities automatically. It needs Python 3.11+.
- Without it, pass your own `spans=[(start, end), ...]`.
- The probe weights (about 50 KB) ship with the package. The first run downloads Qwen3-4B (about 8 GB). It needs a GPU (about 9 GB in bfloat16) or Apple MPS; on a CPU it is slow.

## Use
```python
from hallustream import Detector

det = Detector.from_pretrained("qwen3-4b")

# Live: the answer token by token; when a sentence ends, its entities arrive with a risk
for ev in det.stream("Tell me a bio of Kiara Advani."):
    print(ev.text, end="")
    for e in ev.entities:
        if e.flagged:
            print(f"  <- {e.text!r} risk {e.risk:.2f}")

# After the fact: an answer Qwen3-4B wrote to this prompt
report = det.score("Tell me a bio of Kiara Advani.", answer)
print(report.highlight("ansi"))  # or "plain" / "html"
```

Command line:
```bash
hallustream "Tell me a bio of Kiara Advani."
```

SHARS (ICML 2026) with the probe as its detector: see `hallustream/integrations/shars.py`.

## Thresholds
An entity is flagged when its risk is at least the threshold. Both presets were chosen on 432 dev people. The rates below were then measured on data never used for any choice: 600 people's biographies and 496 texts about other subjects (judge-labeled entities).

| Preset | Threshold | False facts caught | True facts flagged | Flagged facts that are false |
|---|---|---|---|---|
| `balanced` (default) | 0.738 | 72% / 62% | 17% / 14% | 87% / 84% |
| `high_precision` | 0.823 | 62% / 49% | 11% / 8% | 90% / 88% |

The first number in each cell is for biographies, the second for other topics. A number between 0 and 1 also works, e.g. `threshold=0.6`.

## How well it works in this package
Measured with the shipped code (`modal_jobs/library_check.py`, summarized by `research/library_report.py`):

- **Same scores as the research.** On 769 judged entities in 60 texts, the library's risk matches the evaluated detector's to within 0.0001.
- **Finding entities by itself (GLiNER).** These scores were measured on held-out text. A score is AUROC: how often a made-up entity scores higher than a true one, where 0.5 is chance.

| | Texts | Judged entities found by GLiNER | AUROC |
|---|---|---|---|
| Biographies (SHARS test people, greedy) | 150 | 92% of false, 94% of true | **0.876** [0.853, 0.898] |
| Other topics (places, organizations, events, works, species) | 150 | 90% of false, 89% of true | **0.827** [0.781, 0.864] |

  GLiNER also finds entities the judge did not label (such as roles and generic phrases). Those are scored but not part of these numbers.
- **Speed.** Measured on one L4 GPU with greedy decoding over 1,579 tokens (plain transformers loop):

| | ms per token | vs plain decoding |
|---|---|---|
| Plain decoding | 59.6 | — |
| With the live probe | 61.8 | +3.7% |
| With the probe and GLiNER at each sentence end | 62.5 | +5.0% |

## Limits
- **Its own model's text only.** The probe knows what Qwen3-4B knows. On other models' text it is much weaker (0.69 AUROC on Gemini-written biographies), so do not use it to check other models.
- **One model.** The bundled weights are for Qwen3-4B-Instruct-2507. Another model needs its own probe ([training code](../src/hallustream/research/probe_train.py)).
- **It flags, it does not fix.** A low risk is not proof of truth, and the model cannot add facts it does not know. Retrieval does that when a source is available.
- **What it skips.** Sentences where the model says it does not know (e.g. "there is no widely known public figure named …") are not scored, and neither are bare pronouns.
- **Language.** It has been tested on English only.
