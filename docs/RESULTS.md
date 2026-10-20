# Results

All numbers are generated from logged runs (`results.json` / `gate_results.json`), not typed by hand.

## Reproduction check (Step 3)
We ran Obeso et al.'s released linear probe (`obalcells/hallucination-probes`, `qwen2_5_7b_linear`) through this project's own pipeline on Qwen2.5-7B-Instruct and compared against the paper's Table 4. Span-level AUROC: each entity is scored by its highest token probability.

| Test set | Paper | Ours (paper-style span search) | Recall @ 10% FPR (ours) | Ours (exact character offsets) | Entities scored (exact) |
|---|---|---|---|---|---|
| LongFact | 0.8383 | 0.8384 | 0.5436 | 0.8387 | 14,416 |
| LongFact++ | 0.8467 | 0.8469 | 0.5532 | 0.8467 | 10,117 |

Our pipeline reproduces the published numbers to within 0.0002 AUROC. The project's exact-offset labeling (used for all training) gives the same AUROC while keeping every labeled entity and always labeling the correct mention of repeated names.
