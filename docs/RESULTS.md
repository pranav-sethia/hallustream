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
Validation span-AUROC of a linear probe (L2 logistic regression, the same method for every layer) on 1,200 train / 800 validation answers. The layer is chosen on validation only.

**Qwen3.5-4B (hybrid: full attention at layers 3, 7, ..., 31; the rest Gated DeltaNet)**: best layer 18

| Layer | 15 | 18 | 19 | 22 | 23 | 26 | 27 | 30 |
|---|---|---|---|---|---|---|---|---|
| Val AUROC | 0.826 | 0.831 | 0.829 | 0.817 | 0.814 | 0.809 | 0.808 | 0.813 |

Baselines on the same validation entities: max surprise 0.623, mean surprise 0.621, max entropy 0.613, mean entropy 0.614.

**Qwen3-4B-Instruct-2507 (standard attention)**: best layer 22

| Layer | 18 | 22 | 26 | 30 | 34 |
|---|---|---|---|---|---|
| Val AUROC | 0.821 | 0.826 | 0.805 | 0.798 | 0.802 |

Baselines on the same validation entities: max surprise 0.609, mean surprise 0.613, max entropy 0.670, mean entropy 0.662.

In both models the signal peaks at about 55–60% of depth, not the paper's default of 95%. In Qwen3.5, full-attention and linear-attention layers at matched depth score within about 0.01 of each other: depth matters more than layer type.
