# Results

All numbers are generated from logged runs (`data/gate/gate_results.json`, `sweep_results.json`), not typed by hand.

## Reproduction check
We ran Obeso et al.'s released linear probe (`obalcells/hallucination-probes`, `qwen2_5_7b_linear`) through this project's own pipeline on Qwen2.5-7B-Instruct and compared against the paper's Table 4. Span-level AUROC: each entity is scored by its highest token probability.

| Test set | Paper | Ours (paper-style span search) | Recall @ 10% FPR (ours) | Ours (exact character offsets) | Entities scored (exact) |
|---|---|---|---|---|---|
| LongFact | 0.8383 | 0.8384 | 0.5436 | 0.8387 | 14,416 |
| LongFact++ | 0.8467 | 0.8469 | 0.5532 | 0.8467 | 10,117 |

With the paper's span search, our pipeline reproduces the published numbers to within 0.0002 AUROC. The project's exact-offset labeling (used for all training) lands within 0.0004 of the paper while keeping every labeled entity and always labeling the correct mention of repeated names.

## Layer sweep
**Setup**
- Every layer of both models, on 1,200 train / 800 validation answers (11,414 validation entities).
- Each candidate gets the same linear probe: L2 logistic regression on standardized features with balanced classes. Its regularization strength (C = 3e-5) was tuned on the best candidates first. The default C = 1 overfit and cost about 0.01 AUROC.
- Candidates are compared with a paired bootstrap that resamples validation questions.
- Everything here uses validation data only.

![Validation AUROC by depth](img/layer_curve.png)

**Shape of the curve**
- In both models the signal rises smoothly from the first layer, at about 0.71–0.72 AUROC.
- It peaks at about 55–60% of depth: **Qwen3.5-4B layer 17 (0.843)** and **Qwen3-4B layer 21 (0.840)**.
- It then falls to about 0.81–0.82 in the last layers.
- There are no isolated spikes.
- In Qwen3.5, the full-attention layers (circled in the chart) sit on the same curve as the Gated DeltaNet layers around them.
- The best confidence baselines on the same entities score only 0.62–0.67: maximum token surprise and maximum entropy.

**Layer combinations** (22 per model)

| Combination type | Best example, Qwen3.5 | vs best single layer |
|---|---|---|
| Neighbors (e.g. 16+17, 20+21) | 16+17: 0.843 | no gain |
| Runs of 3–4 neighbors | 16+17+18: 0.843 | no gain |
| Spread out (e.g. 11+17+23) | 13+17+21: 0.842 | no gain |
| Average of a band of layers | mean(15..19): 0.842 | no gain |
| Middle + last layers | 17+30: **0.847** | +0.004 (interval above 0), but recall at 10% FPR not higher |

- For Qwen3, no combination beat layer 21 beyond noise.
- Whether Qwen3.5 uses layer 17 alone or 17+30 was decided on the full training set (below), by the rule in [EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md).
- Full tables: `results/sweep_report.json`, produced by `research/sweep_report.py`.

## Trained probe
**Training**
- Recipe: Obeso et al.'s annealed loss, which starts as a per-token loss and shifts to a loss on each entity's riskiest token. Entity tokens are weighted 10x. Optimizer: AdamW with linear learning-rate decay.
- Data: all 20,056 training answers.
- Tuned on validation: the paper's loss vs a token-only loss, learning rate, passes over the data, and L2 strength.
  - The annealed loss won in every case, by +0.02–0.025 AUROC.
  - Accuracy plateaued at 8 passes over the data.
  - Learning rate and L2 strength changed AUROC by at most 0.001.
- Layer choice: the rule fixed in [EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md) picked layers 17 + 30 over layer 17 alone for Qwen3.5. On validation, AUROC was +0.008 [0.006, 0.009] higher and recall at 10% FPR was 64.9% vs 63.1%.
- Frozen before test: probe, Platt calibration and thresholds were set on validation only and committed before the test set was scored ([configs/frozen_choices.json](../configs/frozen_choices.json)).

**Test set:** 9,426 answers and 144,676 entities, scored once.

| | Qwen3.5-4B (layers 17+30) | Qwen3-4B (layer 21) |
|---|---|---|
| Entity AUROC (seed 0) | **0.867** | 0.854 |
| AUROC over 5 seeds | 0.8666 ± 0.0002 | 0.8545 ± 0.0002 |
| Best confidence baseline | 0.630 (max surprise) | 0.678 (max entropy) |
| Probe − baseline (95% CI) | **+0.236** [0.230, 0.243] | +0.177 [0.171, 0.183] |
| Recall / precision at the fpr10 threshold | 57.6% / 71.7% (false-alarm rate 9.6%) | 55.9% / 70.2% (10.0%) |
| Recall / precision at the guard threshold | 25.8% / 87.0% (false-alarm rate 1.6%) | 25.1% / 86.2% (1.7%) |
| AUPRC | 0.739 | 0.722 |
| Calibration error (ECE) | 0.016 | 0.012 |
| Ordinary words flagged at the fpr10 threshold | 3.9% | 3.9% |

**Comparisons**
- Qwen3.5 vs Qwen3: +0.012 AUROC [0.011, 0.014], paired bootstrap over test questions.
- Both probes pass the protocol's main-result bar (≥ +0.05 over the best baseline, interval above 0) by a wide margin.

**By test set and by the model that wrote the answer (Qwen3.5 probe)**

| | LongFact | LongFact++ | Llama-3.3-70B | Llama-3.1-8B | Mistral-24B | Qwen2.5-7B | Gemma-2-9B |
|---|---|---|---|---|---|---|---|
| AUROC | 0.859 | 0.878 | 0.859 | 0.873 | 0.868 | 0.883 | 0.880 |

**Notes**
- Test AUROC is about 0.018 below validation (0.867 vs 0.885). Validation questions come from the same pool as training; test questions are separate.
- The paper's own Qwen2.5-7B linear probe scored 0.838 / 0.847 on these test sets. Its setting is different (one model probing its own answers), so it is context rather than a head-to-head comparison.
- The guard's stop rate on whole answers needs full token streams. It will be measured with the streaming library.

## Own-answer gate: biographies
**Question:** the probe was trained on text written by other models. Does it catch hallucinated entities in biographies the model writes itself? The setup and pass rule were fixed in advance in [EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md).

**Fact checker (checked against FActScore's human labels; 143 reported people, kept apart from the 40 used to pick settings):**
- Judge: `gemini-3.1-flash-lite` (Batch API), with the person's Wikipedia page as evidence.
- Labeling the humans' own facts: balanced accuracy **0.890** (bar 0.80), on 11,049 facts.
- FActScore with the judge's own fact splitting, compared with the human score:

| | Human | Judge | Error (bar 0.03) |
|---|---|---|---|
| InstructGPT | 0.420 | 0.397 | 0.023 |
| ChatGPT | 0.588 | 0.575 | 0.013 |
| PerplexityAI (reported only) | 0.811 | 0.745 | 0.067 |

- Entity labels vs facts the humans marked: balanced accuracy 0.913 (5,669 entities).

**Gate:** 500 FActScore people, `Tell me a bio of {name}.`, greedy decoding. Entity AUROC is Not Supported vs Supported, using the mean of the 5 frozen probe seeds. The 95% intervals come from a bootstrap over people.

| | Qwen3-4B (layer 21) | Qwen3.5-4B (layers 17+30) |
|---|---|---|
| Biographies scored (others said they did not know the person) | 404 | 485 |
| Entities (share Not Supported) | 4,478 (56%) | 7,579 (75%) |
| **Probe AUROC** | **0.828** [0.809, 0.847] | 0.804 [0.781, 0.825] |
| Frozen baseline AUROC | 0.759 (max entropy) | 0.776 (max surprise) |
| Probe − baseline | **+0.069** [0.045, 0.093] | +0.028 [0.008, 0.047] |
| People never seen in probe training | 0.839 (161 people) | 0.814 (192 people) |
| People seen in probe training | 0.821 (243 people) | 0.798 (293 people) |
| Same 398 people for both models | 0.828 | 0.813 |

**Result: passes** (Qwen3-4B AUROC 0.828, bar 0.75). Next comes detect→fix on Qwen3-4B.

**Notes**
- People the probe saw in training do not score higher. The unseen people actually score slightly higher, so the overlap does not inflate the result.
- On the model's own text, the confidence baselines are much stronger (0.76–0.78) than on other models' text (0.63–0.68). The probe's lead shrinks from +0.18–0.24 to +0.03–0.07, and is clearly larger for Qwen3-4B.
- Qwen3.5-4B hit the 512-token limit in 237 of 500 biographies. On the 248 people whose biographies finished, its probe scores 0.712, below the baseline's 0.755. Its finished biographies are mostly short answers about obscure people, with 84% of entities Not Supported. Qwen3-4B on finished biographies scores 0.839.
- Gemini splits biographies into about half as many claims as FActScore's annotators did. Judge FActScores are therefore comparable across our own runs, not with published FActScore numbers.
- Checks: 0.3–0.5% of judged entities could not be found in the text, and teacher forcing reproduced 98.9% of the greedy tokens.
