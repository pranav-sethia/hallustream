# HalluStream

[![CI](https://github.com/pranav-sethia/hallustream/actions/workflows/ci.yml/badge.svg)](https://github.com/pranav-sethia/hallustream/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pranav-sethia/hallustream/blob/main/notebooks/quickstart.ipynb)

HalluStream catches a language model making things up while it writes. A small linear probe reads the model's hidden states and flags the names, dates, places and titles it is probably inventing. It doesn't generate extra text or search the web, and it adds about 5% to generation time.

The [project page](https://pranav-sethia.github.io/hallustream/) replays real runs with the flags appearing live, and has the results and charts.

![Qwen3-4B writing a biography; likely made-up facts are highlighted as each sentence is checked](docs/img/stream.gif)

*A real, unedited run with Qwen3-4B. The flags land on invented facts: her birth date, her mother, a brother, and films and years she was never in.*

## Quick start

```bash
pip install "hallustream[entities] @ git+https://github.com/pranav-sethia/hallustream"
hallustream "Tell me about A Caribbean Mystery."
```

That answers the question and highlights the details that are likely made up. There is also `hallustream chat`, `score` for checking an answer the model already wrote, and `batch` for whole files. From Python:

```python
from hallustream import Detector

det = Detector.from_pretrained("qwen3-4b")
report = det.generate("Tell me a bio of Kiara Advani.")
print(report.summary())
```

It needs a GPU or an Apple Silicon Mac for Qwen3-4B. The [Colab notebook](https://colab.research.google.com/github/pranav-sethia/hallustream/blob/main/notebooks/quickstart.ipynb) runs on the free tier. More in [docs/LIBRARY.md](docs/LIBRARY.md).

## Results

### Inside SHARS

[SHARS](https://arxiv.org/abs/2606.03628) (ICML 2026) cuts hallucinations by writing an answer one sentence at a time. A detector checks each fact, and sentences with doubtful facts get rewritten or redone. SHARS's own detector asks the model about every fact several times. I replaced it with HalluStream and changed nothing else, then ran all 182 of SHARS's test people with Qwen3-4B:

| Per biography | Qwen3-4B alone | SHARS, own detector | SHARS with HalluStream |
|---|---|---|---|
| True claims | 3.2 | 4.1 | **4.0** |
| False claims | 6.9 | 5.3 | **3.0** (−44%) |
| Precision | 29% | 38% | **49%** |
| Time to check one fact | – | 5.7 s | **0.04 s** (~140x less) |
| Time per biography | 7 s | 211 s | **49 s** |

A false claim is one the fact checker couldn't support from the person's Wikipedia page, which is FActScore's convention. The drop in false claims is −2.37 per person (95% interval −2.96 to −1.80), precision rises by 0.083 (0.038 to 0.132), and true claims don't change. Counting only specific details like names, places, dates and numbers, false ones drop 49%.

![False claims per biography vs detection time per fact](docs/img/shars_tradeoff.png)

### Compared with the paper it builds on

The probe comes from [Obeso et al. (2025)](https://arxiv.org/abs/2509.03531). Before building anything new I reproduced their published numbers, then changed two things: I read a middle layer instead of one near the end, and I tuned the training on validation data. Then I adapted the probe to the model's own writing.

| | The paper | HalluStream |
|---|---|---|
| The paper's own benchmark (Qwen2.5-7B, its data and test sets), AUROC | 0.839 / 0.847 | **0.894 / 0.897** |
| Made-up details caught at a 10% false-alarm rate, same benchmark | 54% / 55% | **68% / 69%** |
| The paper's best method, which fine-tunes the model with LoRA | 0.895 / 0.896 | matched, with the model left unchanged |
| Qwen3-4B's own biographies of 600 new people, AUROC | 0.776 | **0.854** |
| False claims per biography inside SHARS | 4.70 | **2.96** |

The two numbers per cell are the LongFact and LongFact++ test sets. On the model's own writing about 496 subjects that aren't people (places, events, organizations, works, species), HalluStream scores 0.834 against 0.668 for the model's own confidence.

### An example

Adil Rami is a French footballer who won the 2018 World Cup. Asked about him, Qwen3-4B made up 10 details, including a birth date, a birthplace and five clubs he never played for. SHARS with its own detector still kept three of the fake clubs. With HalluStream it kept none. The probe flagged the fake birth date at 98% risk and the fake clubs at 60–95%.

## How it was tested

Each test's settings and pass bars were written down in [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) before it was run. Choices were made on separate dev data, and each test set was scored once. None of the test people appear in the probe's training data.

Labels come from a fact checker (Gemini with the subject's Wikipedia page). Against FActScore's human labels it gets 89% balanced accuracy on 11,049 facts, and Claude re-checking 150 claims blind agreed with it 91% of the time. The details are in [docs/RESULTS.md](docs/RESULTS.md).

## Limits

The probe needs the model's hidden states, so it works on open-weight models, and each model needs its own probe. It holds back what the model would make up but can't add facts the model doesn't know; retrieval does that when a source is available. The SHARS test is on biographies, the standard FActScore benchmark. My fact checker is stricter than the GPT-5 setup in the SHARS paper, so absolute precision is lower than theirs, but every method here is judged by the same checker.

## Repository

- `src/hallustream/`: the library (detector, probe, entity finder, CLI, SHARS plug-in, bundled weights)
- `src/hallustream/research/`: data, training, evaluation, the fact checker and the analysis scripts
- `modal_jobs/`: the GPU and batch jobs, run on Modal
- `configs/`: frozen settings and fixed lists of people
- `notebooks/`: the Colab quickstart
- `tests/`: unit tests, run in CI on Python 3.10 and 3.12

Data sources and licenses are in [docs/DATA.md](docs/DATA.md).

## Credits

- Obeso et al., [Real-Time Detection of Hallucinated Entities in Long-Form Generation](https://arxiv.org/abs/2509.03531) (2025): the probe method and training data
- Li et al., [Building Reliable Long-Form Generation via Hallucination Rejection Sampling](https://arxiv.org/abs/2606.03628) (ICML 2026): SHARS
- Min et al., [FActScore](https://arxiv.org/abs/2305.14251) (2023): the people and human labels

## License

Apache-2.0
