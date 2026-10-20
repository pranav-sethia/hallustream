# Changelog

## Unreleased
- Every-layer sweep for both models with tuned regularization: the signal rises smoothly to a peak at about 55-60% depth (Qwen3.5 layer 17, Qwen3 layer 21), then falls off. Main feature caches re-extracted at the chosen layers.
- Protocol: added the layer-choice rule for Step 5 (layer 17 vs layers 17 + 30); training now uses all five source models (earlier draft: three license-clean sources; licenses don't restrict this non-commercial project); answers to reworded copies of test questions are removed (7 answers).
- Reproduced Obeso et al. Table 4 (Qwen2.5-7B linear probe) to within 0.0002 AUROC.
- Project setup: package skeleton, tests for hidden-state indexing and KV-cache decoding, evaluation protocol, Modal smoke job, CI.
