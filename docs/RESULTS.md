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
