# HalluStream

**A near-free hallucination detector for open LLMs.** A small linear probe reads a model's hidden states while it writes and flags the names, dates, places and titles it is probably making up. It generates no extra text and needs no web search.

## Results
**Inside SHARS (ICML 2026), the probe beats SHARS's own detector.** [SHARS](https://arxiv.org/abs/2606.03628) writes sentence by sentence: it splits each sentence into facts, asks a detector which facts are made up, and keeps, rewrites or redoes the sentence. Its detector, HalluSE, asks the model a question about every fact and samples several answers. We replaced HalluSE with the probe and changed nothing else. Results on 60 held-out FActScore people, with Qwen3-4B, pre-registered:

| Per biography | Plain writing | SHARS + HalluSE | **SHARS + probe** |
|---|---|---|---|
| True claims | 3.1 | 4.0 | **4.0** |
| False claims | 7.0 | 5.2 | **3.3** (−36%) |
| Precision | 28% | 38% | **47%** |
| Detection time per fact | – | 5.8 s | **0.05 s** (~110× less) |
| Total time per biography | 8 s | 232 s | **60 s** |

False claims: −1.87 per person, 95% interval [−2.75, −0.95].

**The detector on its own** (AUROC: how often a made-up fact scores higher than a true one; 0.5 is chance):

| | Probe | Model's own confidence |
|---|---|---|
| Other models' long answers on many topics (144,676 held-out entities), Qwen3.5-4B | **0.867** | 0.630 |
| Same, Qwen3-4B | **0.854** | 0.678 |
| Qwen3-4B's own biographies, 600 untouched people (probe adapted to its own writing) | **0.854** | 0.690 |
| Qwen3-4B's own writing about 496 other subjects (places, organizations, events, works, species) | **0.834** | 0.668 |

Example from the test (Mike Trivisonno):
- SHARS + HalluSE kept "He played college basketball at the University of California, Los Angeles. Mike Trivisonno played in the NBA for the Boston Celtics." (10 false claims in all).
- SHARS + probe wrote only "Mike Trivisonno is an American."

All numbers, with intervals and caveats, are in [docs/RESULTS.md](docs/RESULTS.md).

## How it works
1. **Probe.** A logistic classifier on one middle layer of the model (layer 21 of Qwen3-4B, chosen by a sweep over every layer). It scores each entity's tokens. It was trained with the recipe and labeled data of Obeso et al. (2025): about 20,000 long answers on topics from the Apollo 11 landing to corporate law, written by five other models, with every entity marked supported or not by web search.
2. **Adapted to the model's own writing (v3).** Qwen3-4B wrote biographies of 432 separate people, an automatic fact checker labeled them, and a regularized linear term on the same hidden state was added to the probe. It was then checked once on 600 people never used before.
3. **Fact checker.** Gemini 3.1 Flash-Lite reads the subject's Wikipedia page and labels every claim. It agrees with FActScore's human labels on 89% of 11,049 facts (balanced accuracy).
4. **SHARS integration.** SHARS's code is vendored unchanged in [`third_party/shars`](third_party/shars) (MIT). The probe plugs in as an uncertainty estimator ([`modal_jobs/shars.py`](modal_jobs/shars.py)): one forward pass per sentence instead of a question and several sampled answers per fact.

## How it was evaluated
- Every test's settings, metrics and pass bars were committed in [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) before its outputs existed. Amendments are dated there with their reasons.
- Settings were chosen on separate dev people, and each test set was scored once. Paired bootstrap intervals are over people.
- Leak checks: none of the SHARS test people appear in the probe's training data or in v3's fitting data.
- Ideas that did not work are reported too: detect-and-rewrite with retrieval, the probe as a training reward, and a small model checking frontier models' text (see "Ideas tried and stopped" in RESULTS).

## Limits
- It needs the model's hidden states, so it works on open-weight models, and each model needs its own probe.
- It holds back what the model would make up; it cannot add knowledge the model lacks. Retrieval does that when a source is available.
- The SHARS test is on biographies (FActScore, the standard benchmark). The detector itself was also tested on Qwen3-4B's writing about 496 other subjects (0.834 vs 0.668), but SHARS was not run on them.
- The SHARS test has 60 people. The probe's lead over the model's own token surprise inside SHARS is not statistically established.
- Our judge is stricter than SHARS's GPT-5 FActScore setup, so absolute precision is lower than in their paper. All arms share the same judge.

## Repository
- `src/hallustream/research/`: data, probe training, evaluation, the fact checker (`factcheck/`), SHARS analysis
- `src/hallustream/fix/`: scoring text with the probe (`score.py`) and entity spans
- `modal_jobs/`: GPU and batch jobs on Modal (feature extraction, training, biographies, judge batches, SHARS runs)
- `configs/`: frozen choices, detector weights and fixed people lists
- `tests/`: unit tests; CI runs ruff, pyright and pytest

Data sources and licenses: [docs/DATA.md](docs/DATA.md). A Python library and a demo are next.

## Builds on
- Obeso et al., [Real-Time Detection of Hallucinated Entities in Long-Form Generation](https://arxiv.org/abs/2509.03531) (2025): the probe method and training data
- Li et al., [Building Reliable Long-Form Generation via Hallucination Rejection Sampling](https://arxiv.org/abs/2606.03628) (ICML 2026): SHARS
- Min et al., [FActScore](https://arxiv.org/abs/2305.14251) (2023): people and human labels

## License
Apache-2.0
