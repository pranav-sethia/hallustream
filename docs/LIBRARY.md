# Using HalluStream

HalluStream flags the specific details (names, dates, places, numbers, titles, roles) that Qwen3-4B-Instruct-2507 is probably making up, using the model's own hidden states. It doesn't generate anything extra or search the web.

You might use it to warn readers about doubtful details in an app built on an open model, or to look something up only when the model is unsure. Researchers can get per-detail risks for whole datasets, or use it as a cheap detector inside decoding methods like SHARS. Or just run one command and watch what the model makes up as it writes.

## Install

```bash
pip install "hallustream[entities] @ git+https://github.com/pranav-sethia/hallustream"
```

The `[entities]` extra adds [GLiNER](https://github.com/urchade/GLiNER), which finds the details automatically and needs Python 3.11 or newer. Without it you can pass your own `spans=[(start, end), ...]` to `score()`.

The probe weights (about 50 KB) ship with the package. The first run downloads Qwen3-4B, about 8 GB. You'll want a GPU with about 9 GB free or an Apple Silicon Mac; it runs on a CPU, but slowly.

## Command line

| Command | What it does |
|---|---|
| `hallustream "Tell me about Lake Tahoe."` | Answers, printing each sentence once it has been checked |
| `hallustream chat` | The same, as a conversation |
| `hallustream score "Tell me about Lake Tahoe." answer.txt` | Checks an answer the model already wrote (a file or stdin) |
| `hallustream batch prompts.jsonl -o results.jsonl` | Checks a file of prompts or answers and writes one JSON line each |

Here is a real run in plain-text mode. In a terminal, the flagged details show up in red.

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

Ten of those twelve are wrong or unsupported: the birth date, the mother, an invented brother, and films and years she was never in.

Useful options are `--threshold balanced|strict|0.8`, `--json` for one JSON object instead of text (`ask` and `score`), `--max-new-tokens` (default 400), `--no-color` and `--device`.

For `batch`, give it JSON lines with a `"prompt"` (or `"messages"`), or plain text with one prompt per line. Lines that already have an `"answer"` are checked as they are; the rest are answered first. Every input field is copied to the output, so your ids carry through, and the output adds `text`, `threshold`, `n_flagged` and `entities`.

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

In a notebook a `Report` shows up as highlighted text with its summary. Every method also takes `messages=[...]` instead of a prompt, and a `threshold=`.

A few ways to use it:

- Show a warning next to flagged details: `if report.flagged:` render `report.highlight("html")`.
- Stop and look something up: break out of `stream()` at the first flagged detail and answer from a search instead.
  ```python
  for ev in det.stream(question):
      if any(e.flagged for e in ev.entities):
          break
  ```
- Clean up generated data: run `hallustream batch` on the samples and keep the lines with `"n_flagged": 0`.
- Use it inside SHARS: the probe plugs in as SHARS's detector (`hallustream/integrations/shars.py`); the results are in the main README.

## Reading the output

It checks specific details: names, places, titles, dates, numbers, and roles such as "actress". Generic lowercase words like "region", "book" or "team" are skipped, because the fact checker almost never treats them as claims. On the texts below, skipping them removed 16–19% of flags, kept 96–99% of the false details that were found, and left AUROC where it was (0.878 / 0.827; see [`research/entity_rule.py`](../src/hallustream/research/entity_rule.py)). Sentences where the model says it doesn't know someone ("there is no widely known public figure named …") aren't checked, and neither are bare pronouns.

Risk is the probe's estimate of the chance that a detail is made up, where "made up" means a fact checker reading the subject's Wikipedia page found it unsupported. On text it was never tuned on, risk reads as a rough probability ([`research/calibration.py`](../src/hallustream/research/calibration.py)):

| Stated risk | Actually false: 600 people / 496 topics |
|---|---|
| 90–100% | 93% / 92% |
| 70–80% | 67% / 67% |
| 0–10% | 8% / 7% |

On average the stated risk is within 3–4 points of the real rate, running a little high in the middle of the range.

A detail is flagged when its risk is at or above the threshold. Both presets were chosen on 432 dev people, and the rates below were measured afterwards on data never used for any choice: biographies of 600 people and texts about 496 other subjects.

| Preset | Threshold | False details caught | True details flagged | Flags that are right |
|---|---|---|---|---|
| `balanced` (default) | 0.738 | 72% / 62% | 17% / 14% | 87% / 84% |
| `strict` | 0.823 | 62% / 49% | 11% / 8% | 90% / 88% |

The first number in each cell is for biographies and the second for other topics. Any number between 0 and 1 works as a threshold too.

## How well the package works

These were measured with the shipped code (`modal_jobs/library_check.py`, summarized by `research/library_report.py`).

The library gives the same risk as the research code to within 0.0001, checked on 769 judged details in 60 texts. When it finds the details itself with GLiNER, it scores as follows on held-out text (AUROC is how often a made-up detail scores higher than a true one; 0.5 is chance):

| | Texts | Judged details GLiNER finds | AUROC |
|---|---|---|---|
| Biographies (SHARS test people) | 150 | 92% of false, 94% of true | **0.876** (0.853 to 0.898) |
| Other topics (places, organizations, events, works, species) | 150 | 90% of false, 89% of true | **0.827** (0.781 to 0.864) |

Speed, measured on one L4 GPU with greedy decoding over 1,579 tokens:

| | ms per token | vs plain decoding |
|---|---|---|
| Plain decoding | 59.6 | – |
| With the live probe | 61.8 | +3.7% |
| With the probe and GLiNER at each sentence end | 62.5 | +5.0% |

## Limits

The probe reads what Qwen3-4B itself knows, so it's built to check Qwen3-4B's own writing, not other models'. The bundled weights are for Qwen3-4B-Instruct-2507, and another model needs its own probe ([training code](../src/hallustream/research/probe_train.py)).

It flags; it doesn't fix. A low risk isn't proof that something is true, and the model can't add facts it doesn't know. Retrieval does that when you have a source. It also scores details rather than whole claims, so a wrong claim built only from correct names can get through.

It was measured on single questions in English. Later answers in a `chat` conversation are scored the same way, but they weren't part of any measurement.
