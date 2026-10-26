# Using HalluStream

HalluStream flags the specific details (names, dates, places, numbers, titles, roles) that Qwen3-4B-Instruct-2507 is probably making up, from its own hidden states. It adds no extra generation and needs no search.

## Who it is for
- **People building on an open model** (transformers): warn readers about doubtful details, or look something up only when the model is unsure, at about 5% extra generation time.
- **Researchers**: per-detail risks for whole datasets, a cheap detector for decoding methods such as SHARS, and a calibrated baseline to compare against.
- **Anyone curious**: one command shows what the model is making up as it writes.

## Install
```bash
pip install "hallustream[entities] @ git+https://github.com/pranav-sethia/hallustream"
```
- `[entities]` adds [GLiNER](https://github.com/urchade/GLiNER) to find details automatically. It needs Python 3.11+.
- Without it, pass your own `spans=[(start, end), ...]` to `score()`.
- The probe weights (about 50 KB) ship with the package. The first run downloads Qwen3-4B (about 8 GB). It needs a GPU (about 9 GB in bfloat16) or Apple MPS; on a CPU it is slow.

## Command line
| Command | What it does |
|---|---|
| `hallustream "Tell me about Lake Tahoe."` | Answers, with each sentence printed once it has been checked |
| `hallustream chat` | The same, in a conversation |
| `hallustream score "Tell me about Lake Tahoe." answer.txt` | Checks an answer the model already wrote (file or stdin) |
| `hallustream batch prompts.jsonl -o results.jsonl` | Checks a file of prompts or answers and writes one JSON line each |

A real run (plain-text mode; in a terminal the flagged details are red):
```
$ hallustream "Tell me a bio of Kiara Advani."
Kiara Advani is an Indian actress, model, and [[businesswoman]] who has made a significant mark
in the Indian film industry, particularly in Bollywood. Born on [[May 12, 1993]], in [[Mumbai]],
Maharashtra, she is the daughter of renowned Bollywood actor and producer Rajesh Advani and
actress [[Sushmita Dev]]. ...

12 of 21 details are likely made up:
   98%  Sushmita Dev
   98%  May 12, 1993
   97%  Rani
   96%  2014
   95%  2017
   94%  2018
   94%  actor and producer
   93%  Bajirao Mastani
   90%  businesswoman
   86%  2013
   82%  Aryan Advani
   76%  Mumbai
Risk is the estimated chance a detail is made up; flagged from 74%.
```
How the flags hold up:
- Ten of the twelve are wrong or unsupported: the birth date, the mother, an invented brother, and films and years she was never in.
- "Mumbai" is a false alarm.
- "2018" sits in a sentence that was cut off before it made a claim.
- It misses the invented father, "Rajesh Advani".

Options:
- `--threshold balanced|strict|0.8`
- `--json`: one JSON object instead of text (`ask` and `score`)
- `--max-new-tokens` (default 400)
- `--no-color`, `--device`

Batch input is JSON lines, each with a `"prompt"` (or `"messages"`), or one prompt per line:
- Items with an `"answer"` are checked as they are.
- The other items are answered first.
- Every input field is kept in the output, so ids carry through. The output adds `text`, `threshold`, `n_flagged` and `entities`.

## Python
```python
from hallustream import Detector

det = Detector.from_pretrained("qwen3-4b")  # Qwen3-4B-Instruct-2507 + the bundled probe

report = det.generate("Tell me a bio of Kiara Advani.")  # write an answer and check it
report = det.score("Tell me a bio of Kiara Advani.", answer)  # check one it already wrote
print(report.summary())  # the list above
print(report.highlight("ansi"))  # or "plain" ([[...]]) / "html"
report.flagged  # [Entity(text, start, end, risk, flagged), ...]
report.to_dict()  # JSON-ready

for ev in det.stream("Tell me a bio of Kiara Advani."):  # live
    print(ev.text, end="")  # new text as it is written
    ev.entities  # details of sentences that just finished, with their risk
```
In a notebook, a `Report` displays as highlighted text with its summary. Every method takes `messages=[...]` instead of a prompt, and `threshold=`.

**Recipes**
- *Warn readers:* `if report.flagged:` show a warning next to the flagged details (`report.highlight("html")`).
- *Stop early and look it up:* break out of `stream()` at the first flagged detail and answer from a search instead.
  ```python
  for ev in det.stream(question):
      if any(e.flagged for e in ev.entities):
          break
  ```
  When a source exists, use it. In our tests, giving the model the subject's Wikipedia page took precision from 18% to 96% (RESULTS, "Ideas tried and stopped"). That is an upper bound, because the fact checker read the same page.
- *Filter generated data:* `hallustream batch` the samples, then keep the lines with `"n_flagged": 0`.
- *SHARS:* the probe plugs in as SHARS's detector (`hallustream/integrations/shars.py`); results are in the main README.

## Reading the output
**What gets checked.** Specific details: names, places, titles, dates, numbers, and roles such as "actress".
- Generic lowercase words ("region", "book", "team") are skipped. The fact checker almost never treats them as claims.
- On the texts below, skipping them removed 16–19% of flags and kept 96–99% of the false details that were found. AUROC stayed the same (0.878 / 0.827). See [`research/entity_rule.py`](../src/hallustream/research/entity_rule.py).
- Sentences where the model says it does not know (for example "there is no widely known public figure named …") are not checked, and neither are bare pronouns.

**Risk** is the probe's estimate of the chance that a detail is made up. "Made up" means a fact checker reading the subject's Wikipedia page found it unsupported.

On held-out text, risk reads as a rough probability ([`research/calibration.py`](../src/hallustream/research/calibration.py)):

| Stated risk | Actually false: 600 people / 496 topics |
|---|---|
| 90–100% | 93% / 92% |
| 70–80% | 67% / 67% |
| 0–10% | 8% / 7% |

The average gap between stated risk and the actual rate is 3–4 points. Stated risk runs a little high in the middle of the range.

**Flagged** means risk at or above the threshold. Both presets were chosen on 432 dev people. The rates below were then measured on data never used for any choice: 600 people's biographies and 496 texts about other subjects (judge-labeled entities).

| Preset | Threshold | False details caught | True details flagged | Flagged details that are false |
|---|---|---|---|---|
| `balanced` (default) | 0.738 | 72% / 62% | 17% / 14% | 87% / 84% |
| `strict` | 0.823 | 62% / 49% | 11% / 8% | 90% / 88% |

The first number in each cell is for biographies, the second for other topics. Any number between 0 and 1 also works.

## How well it works in this package
Measured with the shipped code (`modal_jobs/library_check.py`, summarized by `research/library_report.py`):

- **Same scores as the research.** On 769 judged entities in 60 texts, the library's risk matches the evaluated detector's to within 0.0001.
- **Finding details by itself (GLiNER).** Measured on held-out text, before the specific-details rule above (which leaves AUROC unchanged). AUROC is how often a made-up detail scores higher than a true one; 0.5 is chance.

| | Texts | Judged entities found by GLiNER | AUROC |
|---|---|---|---|
| Biographies (SHARS test people, greedy) | 150 | 92% of false, 94% of true | **0.876** [0.853, 0.898] |
| Other topics (places, organizations, events, works, species) | 150 | 90% of false, 89% of true | **0.827** [0.781, 0.864] |

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
- **Misses.** At the default threshold it catches about two-thirds of false details. It scores details, not the claims around them, so a wrong claim built from correct names can pass. For example, it did not flag "larger than the entire country of France" about the Great Barrier Reef.
- **Tested on single questions.** Answers later in a `chat` conversation are scored the same way, but they were not part of any measurement.
- **Language.** It has been tested on English only.
