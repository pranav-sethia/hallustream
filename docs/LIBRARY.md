# Using HalluStream

HalluStream flags the specific details (names, dates, places, numbers, titles, roles) that Qwen3-4B-Instruct-2507 is probably making up, using the model's own hidden states. It doesn't generate anything extra or search the web.

You might use it to warn readers about doubtful details in an app built on an open model, or to look something up only when the model is unsure. Researchers can get per-detail risks for whole datasets, or use it as a cheap detector inside decoding methods like SHARS. Or just run one command and watch what the model makes up as it writes.

## Install

```bash
pip install "hallustream[entities]"
```

The `[entities]` extra adds [GLiNER](https://github.com/urchade/GLiNER), which finds the details automatically. Without it you can pass your own `spans=[(start, end), ...]` to `score()`. Python 3.10 or newer.

The probe weights (about 150 KB) ship with the package. The first run downloads Qwen3-4B, about 8 GB. You'll want a GPU with about 9 GB free or an Apple Silicon Mac; it runs on a CPU, but slowly.

## Command line

| Command | What it does |
|---|---|
| `hallustream "Tell me about Lake Tahoe."` | Answers, printing each sentence once it has been checked |
| `hallustream chat` | The same, as a conversation |
| `hallustream score "Tell me about Lake Tahoe." answer.txt` | Checks an answer the model already wrote (a file or stdin) |
| `hallustream batch prompts.jsonl -o results.jsonl` | Checks a file of prompts or answers and writes one JSON line each |

Here is a real run in plain-text mode, on an M2 MacBook (shortened in the middle). In a terminal, the flagged details show up in red.

```
$ hallustream "Tell me a bio of Kiara Advani."
Kiara Advani is an Indian actress, model, and [[businesswoman]] who has made a significant mark
in the Indian film industry, particularly in Bollywood. Born on [[May 10, 1993]], in Mumbai,
Maharashtra, she is the daughter of renowned Bollywood actor and producer [[Rajesh Advani]] and
actress and former model [[Sushmita Dev]]. ...

14 of 25 details are likely made up:
   99%  Sushmita Dev
   97%  May 10, 1993
   95%  The Sky Is Pink
   93%  Kabir Singh
   93%  Dil Dhadakne Do
   93%  2013
   92%  Dhoom 3
   92%  Ae Dil Hai Mushkil
   89%  businesswoman
   86%  2019
   83%  Gully Boy
   82%  2015
   80%  Rajesh Advani
   77%  2016
Risk is the estimated chance a detail is made up; flagged from 76%.
```

Checked against her Wikipedia article, ten of the fourteen are wrong: the birth date, both parents, the debut year, "businesswoman", and five films she was never in. One is a false alarm: she did star in Kabir Singh. The other three are release years of the films named. It missed one invented film, Bajirao Mastani.

Useful options are `--threshold balanced|strict|0.8`, `--json` for one JSON object instead of text (`ask` and `score`), `--max-new-tokens` (default 512), `--no-color` and `--device`.

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

It checks specific details: names, places, titles, dates, numbers, and roles such as "actress". Generic lowercase words like "region", "book" or "team" are skipped, because the fact checker almost never treats them as claims, and so are a capitalized word that only starts a sentence ("The death of …", "As of now") and a lone title ("Mr."). On 600 dev biographies those last two rules removed 2.5% of the details found and 0.1% of the false ones ([`entities.py`](../src/hallustream/entities.py)). Sentences where the model says it doesn't know someone ("there is no widely known public figure named …") aren't checked, and neither are bare pronouns.

Risk is the probe's estimate of the chance that a detail is made up, where "made up" means a fact checker reading the subject's Wikipedia page found it unsupported. On new people and subjects it was never fitted on, risk reads as a probability:

| Stated risk | Actually false: 437 new people / 498 new subjects |
|---|---|
| 90–100% | 96% / 96% |
| 70–80% | 77% / 77% |
| 40–50% | 48% / 42% |
| 0–10% | 11% / 9% |

On average the stated risk is within 2.5 points of the real rate.

A detail is flagged when its risk is at or above the threshold. The presets were set before the new data existed, from out-of-fold scores on the texts the probe was fitted on, to flag 10% (`balanced`) or 5% (`strict`) of true details; the rates below were then measured on the new people and subjects.

| Preset | Threshold | False details caught | True details flagged | Flags that are right |
|---|---|---|---|---|
| `balanced` (default) | 0.764 | 68% / 54% | 12% / 7% | 91% / 90% |
| `strict` | 0.858 | 53% / 39% | 6% / 3% | 94% / 94% |

The first number in each cell is for people and the second for other subjects. Any number between 0 and 1 works as a threshold too.

## How well the package works

These were measured with the shipped code on the same new people and subjects (`modal_jobs/library_check.py`, summarized by `research/library_report.py`).

The library gives the same scores as the research code to within 0.0003 (risk within 0.0001), checked on 736 judged details in 60 texts. When it finds the details itself with GLiNER, it scores as follows (AUROC is how often a made-up detail scores higher than a true one; 0.5 is chance):

| | Texts | Judged details GLiNER finds | AUROC |
|---|---|---|---|
| People | 114 | 77% of false, 82% of true | **0.858** (0.813 to 0.898) |
| Other subjects (places, organizations, events, works, species) | 150 | 82% of false, 84% of true | **0.851** (0.814 to 0.886) |

GLiNER misses some details the fact checker judged, more often in these texts about little-known subjects than in an earlier check on FActScore's better-known people (92% of false details found). `stream()` scores the same details as `score()`; their risks differ by at most about 0.02 on Qwen3-4B, because a streamed answer is tokenized as it was generated.

Speed, measured on one L4 GPU with greedy decoding over 1,600 tokens:

| | ms per token | vs plain decoding |
|---|---|---|
| Plain decoding | 60.5 | – |
| With the live probe | 62.1 | +2.7% |
| With the probe and GLiNER at each sentence end | 64.6 | +6.8% |

On an M2 MacBook (16 GB) Qwen3-4B runs at about 100 ms per token in bfloat16 and needs about 10 GB of memory.

## Limits

The probe reads what Qwen3-4B itself knows, so it's built to check Qwen3-4B's own writing, not other models'. The bundled weights are for Qwen3-4B-Instruct-2507, and another model needs its own probe ([training code](../src/hallustream/research/probe_train.py)).

It flags; it doesn't fix. A low risk isn't proof that something is true, and the model can't add facts it doesn't know. Retrieval does that when you have a source. It also scores details rather than whole claims, so a wrong claim built only from correct names can get through.

It was measured on single questions in English. Later answers in a `chat` conversation are scored the same way, but they weren't part of any measurement.
