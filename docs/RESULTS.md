# Results

Every number here comes from logged runs and the scripts named in each section. Each test's settings and pass bars were written in [EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md) before it was run. Choices were made on validation or dev data, and each test set was scored once. Intervals are 95% paired bootstrap intervals (2,000 resamples) over people or answers.

In short: HalluStream beats the paper it builds on, on that paper's own benchmark, and matches the paper's fine-tuned version without changing the model. Plugged into SHARS (ICML 2026) on all 182 of its test people, it cuts false claims by 44% compared with SHARS's own detector, with higher precision and about 140x less detection time. On Qwen3-4B's own writing it scores 0.854 AUROC on 600 new people and 0.834 on 496 other subjects.

## Against the paper on its own benchmark

Obeso et al. released a linear probe for Qwen2.5-7B-Instruct that reads layer 26 of 28, their default of about 0.95 of the depth. I kept everything else from the paper's setting and asked whether my recipe does better.

The model stays frozen. Training uses the paper's own annotations of Qwen2.5-7B's answers: 1,625 answers from the official train split, with the 178-answer validation split for every choice, and the official test split scored once at the end. I picked the layer on validation with the same sweep I used for Qwen3-4B (chart below): layer 18 scored 0.874 against 0.845 for layer 26. Training uses the paper's annealed loss, with the learning rate, number of passes and L2 strength tuned on validation, averaged over 5 seeds.

Both probes are scored on exactly the same details. The paper's probe, run through this pipeline, gives the numbers the paper published.

| | LongFact (14,416 details) | LongFact++ (10,117) |
|---|---|---|
| Paper's released probe (layer 26) | 0.839 (54% caught) | 0.847 (55% caught) |
| My recipe at layer 26 | 0.870 | 0.880 |
| My recipe at layer 18 | **0.894 (68% caught)** | **0.897 (69% caught)** |
| Gain over the paper's probe | **+0.055 (0.049 to 0.062)** | **+0.050 (0.043 to 0.057)** |
| Paper's LoRA probe (fine-tunes the model, their Table 4) | 0.895 (66% caught) | 0.896 (68% caught) |

"Caught" is the share of made-up details flagged at a 10% false-alarm rate. That is about 13 more points than the paper's probe. Roughly +0.03 of the gain comes from the tuned training at the same layer, and +0.017 to +0.024 from reading a middle layer. The result matches the paper's LoRA probe, which changes the model's weights, while this probe leaves the model alone and costs one dot product per token. The LoRA row uses the paper's published values on the same test sets.

Before any of this, I ran the paper's released probe through my own pipeline as a check. It gave 0.8384 / 0.8469 against the published 0.8383 / 0.8467. The code is `modal_jobs/extract.py` (run `main_qwen25`), `modal_jobs/sweep.py`, `modal_jobs/train.py` (variants `qwen25_L18` and `qwen25_L26`) and `research/paper_h2h.py`.

![Validation AUROC of a probe on each layer](img/layer_curve.png)

In both models the signal peaks at about 60–70% of the depth and falls off toward the last layers, which is where the paper reads by default (`research/layer_figure.py`).

## Inside SHARS

[SHARS](https://arxiv.org/abs/2606.03628) (Li et al., ICML 2026) writes a biography one sentence at a time. Each sentence is split into facts, a detector scores every fact, and the sentence is kept, rewritten from its trusted facts, or redone. SHARS's own detector, HalluSE, writes a question about each fact, samples several answers and compares them. I replaced it with HalluStream and left everything else alone: Qwen3-4B-Instruct-2507, with SHARS's code, prompt, rewriting and resampling.

The test covers all 182 of SHARS's FActScore people, none of whom are in the probe's training data. Every biography was checked by the fact checker described below. A false claim is one it couldn't support from the person's Wikipedia page, which is FActScore's convention. Settings were chosen on separate dev people, and the probe and the confidence baselines flag the same share of dev facts.

| Per person (182) | Qwen3-4B alone | SHARS + HalluSE | SHARS + token surprise | SHARS + probe trained the paper's way | SHARS + HalluStream |
|---|---|---|---|---|---|
| True claims | 3.18 | 4.05 | 3.79 | 3.10 | **4.04** |
| False claims | 6.87 | 5.33 | 4.48 | 4.70 | **2.96** |
| Precision (answered) | 0.290 | 0.376 | 0.437 | 0.338 | **0.486** |
| Answered | 92% | 93% | 90% | 90% | 85% |
| True specific details | 1.88 | 2.15 | 1.93 | 1.84 | **2.05** |
| False specific details | 3.95 | 2.74 | 2.18 | 2.77 | **1.41** |
| Detection time per fact (L4) | – | 5.71 s | 0.043 s | 0.035 s | **0.040 s** |
| Time per biography (L4) | 7.2 s | 210.9 s | 52.0 s | 41.2 s | **49.0 s** |

Specific details are claims with a name, place, date or number (`research/shars_test.py`).

Compared with SHARS's own detector, paired over the 182 people:

| | Difference (95% interval) |
|---|---|
| False claims | **−2.37 (−2.96 to −1.80)**, −44% |
| True claims | −0.01 (−0.47 to +0.46) |
| Precision (154 people answered by both) | **+0.083 (+0.038 to +0.132)** |
| False specific details | **−1.33 (−1.64 to −1.01)**, −49% |
| True specific details | −0.10 (−0.31 to +0.11) |

So SHARS with HalluStream makes 44% fewer false claims, keeps the same number of true ones, and has higher precision. Detection is about 140x cheaper (0.040 s against 5.71 s per fact), which makes the whole SHARS run 4.3x faster. I tested the first 60 people before the rest, and the other 122 on their own show the same thing: −2.62 false claims (−3.33 to −1.94) and +0.082 precision (+0.032 to +0.133). HalluStream also beats the model's own token surprise, with −1.52 false claims (−2.13 to −0.92) and +0.049 precision (+0.003 to +0.099).

The adaptation to the model's own writing is what makes the difference here. A probe trained only the paper's way (same layer, the paper's data, no adaptation) is no better than SHARS's own detector. The adapted one removes 1.74 more false claims per person (1.13 to 2.42), keeps 0.94 more true claims (0.57 to 1.34), and has 0.132 higher precision (0.087 to 0.179).

Two checks on these numbers. First, to make sure the fact checker wasn't favouring one method, Claude re-labeled 150 claims blind against the full Wikipedia articles, sampled evenly across methods and labels. It agreed with the fact checker on 91.3% of them (κ 0.83), and on 90% for both SHARS's detector and HalluStream. Where they differed, the fact checker had accepted vague praise, equally for both. Second, to make sure the probe's edge doesn't come from learning the fact checker's habits: the probe trained the paper's way never saw its labels, and on 987 dev facts it still ranks false facts above true ones better than SHARS's detector at its best settings (0.692 against 0.636 AUROC). Repeated sentences are counted once.

## The model's own writing

The paper trains its probe on other models' answers. I adapted it to the model's own writing with a regularized linear term on the layer-21 hidden state, fitted on Qwen3-4B's biographies of 432 dev people and labeled by the fact checker below. The setting was chosen by nested cross-validation on dev. It reads only the hidden state the probe already uses, so it still works while the model writes.

On 600 people that weren't used for anything else (Qwen3-4B answered 477 of them, with 3,727 checked details, 61% unsupported):

| | AUROC |
|---|---|
| Model's own confidence (max entropy) | 0.690 (0.668 to 0.711) |
| Probe trained the paper's way | 0.776 (0.751 to 0.800) |
| HalluStream, adapted | **0.854 (0.834 to 0.873)** |
| Adapted minus the paper's way | **+0.078 (0.064 to 0.091)** |
| Adapted minus confidence | **+0.164 (0.138 to 0.192)** |

It also works beyond biographies. On Qwen3-4B's writing about 496 random Wikipedia subjects that aren't people (4,549 checked details), HalluStream scores 0.834, against 0.777 for the paper's way and 0.668 for the model's confidence (+0.166, 0.138 to 0.193). The subjects are living things, events, organizations, creative works and places, spread over four levels of popularity. The adaptation was fitted only on biographies and still adds +0.056 (0.046 to 0.066) here.

| By category | HalluStream | Model's confidence |
|---|---|---|
| Living things | 0.792 | 0.551 |
| Events | 0.860 | 0.763 |
| Organizations | 0.862 | 0.701 |
| Creative works | 0.793 | 0.624 |
| Places | 0.839 | 0.644 |

## The base probe

The base probe is a linear probe on Qwen3-4B's layer 21, picked by the every-layer sweep. It is trained with the paper's annealed loss, which starts as a per-token loss and shifts to each entity's riskiest token, with entity tokens weighted 10x. It used all 20,056 training answers of the paper's data, with choices tuned on validation and frozen before the test, over 5 seeds. The test set is 9,426 answers written by five other models, with 144,676 labeled details.

| | Value |
|---|---|
| AUROC (5 seeds) | **0.854** ± 0.0002 |
| Best confidence baseline (max entropy) | 0.678 |
| Probe minus baseline | +0.177 (0.171 to 0.183) |
| Recall / precision at a 10% false-alarm rate | 55.9% / 70.2% |
| Calibration error (ECE) | 0.012 |

## The fact checker

All the own-writing and SHARS labels come from one fact checker: Gemini 3.1 Flash-Lite through the Batch API, given the subject's Wikipedia page (the 24 best-matching passages by BM25). The prompts are in `research/factcheck/`.

I validated it before using it. Against FActScore's human labels it gets 0.890 balanced accuracy on 11,049 facts, from 143 people kept apart from the 40 used to pick its settings. Its FActScore scores land within 0.023 of the human ones for InstructGPT and 0.013 for ChatGPT. Gemini 3.5 Flash re-judged 300 details of Qwen3-4B's biographies and 300 of its other-topic texts, agreeing on 0.856 and 0.859 of them, and Claude did the SHARS check above.

## The library

The pip package, command line and Python API are measured in [LIBRARY.md](LIBRARY.md). In short: the same scores as the research code to within 0.0001, 0.876 / 0.827 AUROC when it finds the details itself, calibrated risk, and 5% extra time per generated token.
