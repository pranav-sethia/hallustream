# Changelog

## Unreleased
- SHARS (ICML 2026) with the probe as its detector: on 60 held-out FActScore people, 36% fewer false claims than SHARS's own detector at equal true claims, ~110x less detection time (pre-registered). Earlier steps: dev detector comparison, an end-to-end check on 40 dev people, and a 2-person smoke test of every arm.
- Ideas tried and stopped by their pre-registered rules: detect-and-rewrite with retrieval, the probe as a training reward, a small model checking frontier models' text.
- Detector v3 (probe adapted to Qwen3-4B's own biographies): 0.854 AUROC on 600 untouched people vs 0.690 for the model's confidence. Detector v2 (probe + confidence) and the own-answer gate (0.828) before it.
- Fact checker validated against FActScore's human labels (0.890 balanced accuracy on 11,049 facts).
- Trained and evaluated the probes: Qwen3.5-4B layers 17+30 reach 0.867 test AUROC on 144,676 entities (+0.236 over the best confidence baseline; +0.012 over Qwen3-4B). Choices were frozen on validation before test.
- Every-layer sweep for both models with tuned regularization: the signal rises smoothly to a peak at about 55-60% depth (Qwen3.5 layer 17, Qwen3 layer 21), then falls off. Main feature caches re-extracted at the chosen layers.
- Protocol: added the layer-choice rule for the final probe (layer 17 vs layers 17 + 30); training now uses all five source models (earlier draft: three license-clean sources; licenses don't restrict this non-commercial project); answers to reworded copies of test questions are removed (7 answers).
- Reproduced Obeso et al. Table 4 (Qwen2.5-7B linear probe) to within 0.0002 AUROC.
- Project setup: package skeleton, tests for hidden-state indexing and KV-cache decoding, evaluation protocol, Modal smoke job, CI.
