# Results

All numbers are generated from logged runs (`data/gate/gate_results.json`, `sweep_results.json`), not typed by hand.

## Reproduction check (Step 3)
We ran Obeso et al.'s released linear probe (`obalcells/hallucination-probes`, `qwen2_5_7b_linear`) through this project's own pipeline on Qwen2.5-7B-Instruct and compared against the paper's Table 4. Span-level AUROC: each entity is scored by its highest token probability.

| Test set | Paper | Ours (paper-style span search) | Recall @ 10% FPR (ours) | Ours (exact character offsets) | Entities scored (exact) |
|---|---|---|---|---|---|
| LongFact | 0.8383 | 0.8384 | 0.5436 | 0.8387 | 14,416 |
| LongFact++ | 0.8467 | 0.8469 | 0.5532 | 0.8467 | 10,117 |

With the paper's span search, our pipeline reproduces the published numbers to within 0.0002 AUROC. The project's exact-offset labeling (used for all training) lands within 0.0004 of the paper while keeping every labeled entity and always labeling the correct mention of repeated names.

## Layer sweep (Step 4)
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
- Whether Qwen3.5 uses layer 17 alone or 17+30 is decided in Step 5 on the full training set, by the rule in [EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md).
- Full tables: `results/sweep_report.json`, produced by `research/sweep_report.py`.

## Trained probe (Step 5)
**Training**
- Recipe: Obeso et al.'s annealed loss, which starts as a per-token loss and shifts to a loss on each entity's riskiest token. Entity tokens are weighted 10x. Optimizer: AdamW with linear learning-rate decay.
- Data: all 20,056 training answers.
- Tuned on validation: the paper's loss vs a token-only loss, learning rate, passes over the data, and L2 strength.
  - The annealed loss won in every case, by +0.02–0.025 AUROC.
  - Accuracy plateaued at 8 passes over the data.
  - Learning rate and L2 strength changed AUROC by at most 0.001.
- Layer choice: the rule fixed in [EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md) picked layers 17 + 30 over layer 17 alone for Qwen3.5. On validation, AUROC was +0.008 [0.006, 0.009] higher and recall at 10% FPR was 64.9% vs 63.1%.
- Frozen before test: probe, Platt calibration and thresholds were set on validation only and committed before the test set was scored ([configs/step5_frozen.json](../configs/step5_frozen.json)).

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
- The guard's stop rate on whole answers needs full token streams. It is measured with the library in Step 6.
