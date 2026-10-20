# Evaluation protocol (pre-registered)

This file was committed **before** any probe was trained. Its purpose is to fix in advance how results are measured, so they can't be tuned after the fact. Any later change is logged in `CHANGELOG.md` with the reason.

## What is scored
- **Unit:** one labeled entity span (a name, date, number, place or citation), scored by the **highest** per-token risk inside it.
- **Labels:** "Not Supported" and "Insufficient Information" = positive (hallucinated); "Supported" = negative; missing labels are ignored.
- **Own-answer data (Step 7):** "Contradicted" = positive; "Supported" = negative; "Not on the page" is dropped.

## Data splits
- **Annotated data:** `obalcells/longfact-annotations` and `obalcells/longfact-augmented-annotations` (pinned revisions in `src/hallustream/research/data.py`).
- **Test:** the official test splits of both datasets, all source models.
- **Train/validation:** official splits of all five source models (Qwen2.5-7B, Mistral-Small-24B, Gemma-2-9B, Llama-3.1-8B, Llama-3.3-70B).
- Any train/validation answer whose question (normalized: lowercased, whitespace collapsed) appears in the test set is removed, as are answers to reworded copies of test questions (TF-IDF cosine ≥ 0.9, see `docs/DATA.md`).
- **Own-answer data:** split by **topic** into train / validation / test.
- **Nothing is chosen using test data.** Layers, thresholds, calibration and hyperparameters are all chosen on validation.

## Layer choice (validation only)
- An every-layer sweep (2,000 answers, logistic probe with tuned regularization) picked **Qwen3.5-4B layer 17** and **Qwen3-4B layer 21** as the best single layers.
- For Qwen3.5, layers 17 + 30 together scored 0.004 AUROC higher on validation (bootstrap interval above 0). It was selected from about 54 candidates, though, and its recall at 10% false alarms was not higher.
- **Rule, fixed before the Step 5 training:** train the final recipe on both layer 17 (A) and layers 17 + 30 (B). Ship B only if, on validation:
  1. its AUROC is higher than A's, with the paired-bootstrap 95% interval above 0; and
  2. its recall at 10% false alarms is not lower than A's.
  Otherwise ship A.

## Metrics
- **Primary:** AUROC.
- **Also:**
  - recall and precision at the threshold giving 10% false alarms on validation (`fpr10`)
  - AUPRC
  - expected calibration error (ECE)
  - false-alarm rate on ordinary (non-entity) tokens
  - guard stop rate (share of answers the `guard` preset would stop)

## Baselines (same forward pass, same spans)
- Token surprise (negative log-likelihood), entropy, and lowest token probability, each as max and mean over the span. The best variant is chosen on validation.

## Statistics
- **Paired bootstrap** with 1,000 resamples of the difference (probe − baseline), resampling **whole questions** (whole topics for own-answer data).
- **Pass bar for the main result:** difference ≥ 0.05 AUROC, with the 95% interval above 0.
- **Seeds:** 5 training seeds (mean ± spread reported); seed 42 for everything else.

## Gates (decided in advance)
| Gate | Pass condition | If it fails |
|---|---|---|
| Reproduction (paper's Qwen2.5-7B probe through our pipeline) | AUROC within ±0.02 of 0.8383 (LongFact) and 0.8467 (LongFact++) | say "applies", not "reproduces" |
| Leak removal | ≤ 25% of training answers removed | use the LongFact++ test set only |
| Main result | as above | try 2-layer or small-network fallbacks, then report as-is |
| Own-answer | AUROC ≥ 0.70 and κ ≥ 0.6 (50 hand labels) | report as-is; drop "while it writes" claims |
| Guard | stops ≤ 10% of answers | tighten the `guard` threshold |
| Neutral prompt (pasted text) | AUROC drop ≤ 0.03 | make the question field required |

## Fixed prompts
Added here **before** each is first used:
- the neutral default prompt: `Write a detailed, factual answer.`
- the Step 7 entity-extraction and judge prompts
- the P(True) prompt
