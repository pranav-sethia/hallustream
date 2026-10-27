# Results

Every number here is computed from logged runs by the scripts named in each section. Each test's settings and pass rules were committed in [EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md) before its outputs existed, choices were made on validation or dev data only, and each test set was scored once. Intervals are 95% paired bootstrap intervals (2,000 resamples) over people or answers.

**Summary**
- **Beats the paper it builds on, on that paper's own benchmark.** On Qwen2.5-7B with the paper's own training data and test sets, AUROC is 0.894 / 0.897 vs 0.839 / 0.847 for the paper's released probe. That matches the paper's LoRA fine-tuned probe while leaving the model untouched.
- **Cuts false claims inside SHARS (ICML 2026).** On all 182 of SHARS's test people:
  - 44% fewer false claims than SHARS's own detector, with the same number of true claims
  - higher precision
  - about 140× less detection time per fact
- **Works on the model's own writing.** 0.854 AUROC on Qwen3-4B's biographies of 600 untouched people (the paper's recipe: 0.776; the model's own confidence: 0.690), and 0.834 on its writing about 496 other subjects.

## Against the paper on its own benchmark
Obeso et al. released a linear probe for Qwen2.5-7B-Instruct that reads layer 26 of 28 (their default, about 0.95 of depth). This test keeps everything else as in the paper and asks whether this project's recipe does better.

**Setting**
- Model: Qwen2.5-7B-Instruct, frozen.
- Data: the paper's own annotations of Qwen2.5-7B's answers. The official train split (1,625 answers) is for training and the official validation split (178) for every choice. The official test split is scored once.
- Layer: chosen on validation with the same sweep used for Qwen3-4B (chart below). Layer 18 scored 0.874, against 0.845 at layer 26.
- Training: the paper's annealed loss, with learning rate, passes and L2 strength tuned on validation; 5 seeds, averaged.

**Test, on exactly the same details** (the paper's probe was run through this pipeline and matches its published numbers):

| | LongFact (14,416 details) | LongFact++ (10,117) |
|---|---|---|
| Paper's released linear probe (layer 26) | 0.839 (recall 54%) | 0.847 (recall 55%) |
| This recipe at the paper's layer 26 | 0.870 | 0.880 |
| **This recipe, layer 18** | **0.894 (recall 68%)** | **0.897 (recall 69%)** |
| **Gain over the paper's probe** | **+0.055 [0.049, 0.062]** | **+0.050 [0.043, 0.057]** |
| Paper's LoRA probe (fine-tunes the model; its Table 4) | 0.895 (recall 66%) | 0.896 (recall 68%) |

Recall is the share of made-up details caught at a 10% false-alarm rate.

- **Beats the paper's own probe on its own model and test sets** by about +0.05 AUROC, and catches about 13 more points of made-up details at the same false-alarm rate.
- **Where the gain comes from:**
  - about +0.03 from training choices tuned on validation (same layer, same data)
  - +0.017 to +0.024 from reading a middle layer instead of one near the end
- **Matches the paper's LoRA probe without fine-tuning:** the same AUROC and higher recall. The LoRA probe changes the model's weights; this probe leaves the model untouched and costs one dot product per token.
- The LoRA row uses the paper's published values on the same test sets.
- **Reproduction first:** before any new work, the paper's released probe was run through this pipeline. It gave 0.8384 / 0.8469 against the published 0.8383 / 0.8467.
- **Code:** `modal_jobs/extract.py` (run `main_qwen25`), `modal_jobs/sweep.py`, `modal_jobs/train.py` (variants `qwen25_L18`, `qwen25_L26`), `research/paper_h2h.py`.

![Validation AUROC of a probe on each layer](img/layer_curve.png)

In both models the signal peaks at about 60–70% of depth and falls off toward the last layers, where the paper reads by default (`research/layer_figure.py`).

## Inside SHARS: fewer false claims, far cheaper
[SHARS](https://arxiv.org/abs/2606.03628) (Li et al., ICML 2026) writes a biography sentence by sentence. Each sentence is split into facts, a detector scores every fact, and the sentence is kept, rewritten from its trusted facts, or redone. SHARS's own detector, HalluSE, writes a question about each fact, samples several answers and compares them. Here the probe replaces HalluSE and nothing else changes.

**Setting**
- Model: Qwen3-4B-Instruct-2507, with SHARS's code, prompt, rewriting and resampling unchanged.
- People: all 182 of SHARS's FActScore test people. None appear in the probe's training data.
- Labels: every biography is checked by the validated fact checker below. A false claim is one it could not support from the person's Wikipedia page, FActScore's convention.
- Settings were chosen on separate dev people. The probe and the confidence baselines flag the same share of dev facts.

| Per person (182) | Qwen3-4B alone | SHARS + HalluSE | SHARS + token surprise | SHARS + probe trained the paper's way | **SHARS + HalluStream** |
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

**HalluStream vs SHARS's own detector** (paired over 182 people):

| | Difference [95% interval] |
|---|---|
| False claims | **−2.37 [−2.96, −1.80]** (−44%) |
| True claims | −0.01 [−0.47, +0.46] |
| Precision (154 people answered by both) | **+0.083 [+0.038, +0.132]** |
| False specific details | **−1.33 [−1.64, −1.01]** (−49%) |
| True specific details | −0.10 [−0.31, +0.11] |

- **44% fewer false claims and higher precision, with the same number of true claims.**
- **Detection is about 140× cheaper** (0.040 vs 5.71 s per fact), so SHARS runs 4.3× faster end to end.
- **The result replicates.** The first 60 people were tested first. The other 122 alone give false claims −2.62 [−3.33, −1.94] and precision +0.082 [+0.032, +0.133].
- **It also beats the free confidence signal:** false claims −1.52 [−2.13, −0.92], precision +0.049 [+0.003, +0.099].

**This project's probe vs the paper's recipe, inside SHARS:**
- The probe trained the paper's way (same layer, the paper's data, no adaptation) is no better than SHARS's own detector here.
- The probe adapted to the model's own writing (below) removes 1.74 more false claims per person [1.13, 2.42] and keeps 0.94 more true claims [0.57, 1.34]. Precision is +0.132 [+0.087, +0.179].

**Checks on these numbers**
- **The fact checker is fair to every method.** Claude re-labeled 150 claims blind against full Wikipedia articles, sampled equally across methods and labels.
  - It agreed with the fact checker on 91.3% (κ 0.83), and equally for SHARS's detector and HalluStream (90% each).
  - Where they differed, the fact checker had accepted vague praise. That happened equally for both detectors.
- **The probe's advantage does not come from learning the fact checker.** The probe trained the paper's way never saw its labels. On 987 dev facts it still ranks false facts above true ones better than SHARS's detector at its best settings (0.692 vs 0.636 AUROC).
- **Repeated sentences are counted once.**

## The detector on the model's own writing
The paper's probe is trained on other models' answers. HalluStream adapts it to the model's own writing.
- **Method:** a regularized linear term on the layer-21 hidden state, fitted on Qwen3-4B's own biographies of 432 dev people and labeled by the fact checker below. The setting was chosen by nested cross-validation on dev.
- **It reads only the hidden state the probe already uses,** so it still works while the model writes.

**600 untouched people** (Qwen3-4B answered 477; 3,727 checked details, 61% unsupported):

| | AUROC |
|---|---|
| Model's own confidence (max entropy) | 0.690 [0.668, 0.711] |
| Probe trained the paper's way | 0.776 [0.751, 0.800] |
| **HalluStream (adapted)** | **0.854 [0.834, 0.873]** |
| Adapted − paper's way | **+0.078 [0.064, 0.091]** |
| Adapted − confidence | **+0.164 [0.138, 0.192]** |

**Beyond biographies:** Qwen3-4B's writing about 496 random Wikipedia subjects that are not people (4,549 checked details).
- The subjects are living things, events, organizations, creative works and places, spread over four popularity levels.
- **HalluStream 0.834**, against 0.777 for the paper's way and 0.668 for the model's confidence. The difference from confidence is +0.166 [0.138, 0.193].
- The adaptation was fitted only on biographies, yet it carries over: +0.056 [0.046, 0.066] over the paper's way.

| By category | HalluStream | Model's confidence |
|---|---|---|
| Living things | 0.792 | 0.551 |
| Events | 0.860 | 0.763 |
| Organizations | 0.862 | 0.701 |
| Creative works | 0.793 | 0.624 |
| Places | 0.839 | 0.644 |

## The base probe
- **Probe:** a linear probe on Qwen3-4B's layer 21, picked by the every-layer sweep above.
- **Training:**
  - the paper's annealed loss (a per-token loss that shifts to each entity's riskiest token), entity tokens weighted 10×
  - all 20,056 training answers of the paper's data
  - choices tuned on validation and frozen before test
  - 5 seeds
- **Test:** 9,426 answers written by five other models, with 144,676 labeled details.

| | Value |
|---|---|
| AUROC (5 seeds) | **0.854** ± 0.0002 |
| Best confidence baseline (max entropy) | 0.678 |
| Probe − baseline | +0.177 [0.171, 0.183] |
| Recall / precision at a 10% false-alarm rate | 55.9% / 70.2% |
| Calibration error (ECE) | 0.012 |

## The fact checker
Every own-answer and SHARS label comes from one fact checker:
- Gemini 3.1 Flash-Lite (Batch API)
- given the subject's Wikipedia page (BM25 top 24 passages)
- prompts in `research/factcheck/`

It was validated before use:
- **Against FActScore's human labels:** 0.890 balanced accuracy on 11,049 facts (143 people kept apart from the 40 used to pick settings).
- **FActScore scores against the human ones:** off by 0.023 for InstructGPT and 0.013 for ChatGPT (bar 0.03).
- **Spot checks with stronger models:** Gemini 3.5 Flash re-judged 300 details of Qwen3-4B's own biographies and 300 of its other-topic texts (0.856 and 0.859 agreement); Claude did the SHARS check above.

## The library
The shipped package (`pip install`, CLI and Python API) is measured in [LIBRARY.md](LIBRARY.md):
- the same scores as the research code to within 0.0001
- 0.876 / 0.827 AUROC when it finds details by itself
- calibrated risk
- +5% time per generated token
