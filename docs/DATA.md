# Data, models, and licenses

## Annotated data (downloaded at runtime; never re-hosted)
| Dataset | Used for | Notes |
|---|---|---|
| [`obalcells/longfact-annotations`](https://huggingface.co/datasets/obalcells/longfact-annotations) | train / val / test | Test split = the paper's "LongFact" set |
| [`obalcells/longfact-augmented-annotations`](https://huggingface.co/datasets/obalcells/longfact-augmented-annotations) | train / val / test | Test split = the paper's "LongFact++" set |

Both datasets come from Obeso et al., *Real-Time Detection of Hallucinated Entities in Long-Form Generation* ([arXiv 2509.03531](https://arxiv.org/abs/2509.03531)). Their entity labels were produced by Claude 4 Sonnet with web search (84% agreement with human labels in the paper's check).

The datasets carry **no license tag**, so this project never redistributes them. Code downloads them from the source at a pinned revision.

## Which answers are used
- **Training:** the official train splits of both datasets, all five source models. After removing any answer whose question appears in a test set, that leaves 20,056 answers, plus 2,395 for validation (7.7% removed).
- **Reworded-question check:** exact matching can miss rewordings, so `research/leak_check.py` compares every test question with every train/val question (TF-IDF cosine). Of 1,986 test questions, 5 had a train/val match at ≥ 0.9; reading them by hand, all ask about the same subject, so their 7 train/val answers are dropped before training. Pairs below 0.9 are mostly sibling topics that share a prompt template.
- **Test:** the official test splits of both datasets, all five sources: 9,426 answers.
- **Not used:** the `obalcells/hallucination-heads*` datasets. They re-export the same answers, so mixing them in would leak test data into training.

## Why these datasets
These are the only large public datasets with **entity-level** hallucination labels on **long, closed-book** answers written naturally by real models. That is exactly the setting HalluStream targets. We considered and skipped:
- **RAGTruth:** labels faithfulness to a provided document, not factual recall.
- **FAVA training data:** errors are inserted synthetically.
- **HaluEval / ANAH / FELM:** answer- or sentence-level labels, or not English.
- **PsiloQA:** short QA, mostly non-English, labels from GPT-4o against one passage.

## Models
| Use | Model | License |
|---|---|---|
| Main | `Qwen/Qwen3.5-4B` (hybrid: 24 Gated DeltaNet + 8 full-attention layers) | Apache-2.0 |
| Comparison (same size, standard attention) | `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 |
| Reproduction check | `Qwen/Qwen2.5-7B-Instruct` + `obalcells/hallucination-probes` (`qwen2_5_7b_linear`) | Apache-2.0 |
| Tests | `trl-internal-testing/tiny-Qwen3ForCausalLM` | — |

## Own-answer data (Step 7)
- Qwen3 answers questions about Wikipedia topics.
- Claude Haiku 5.5 labels each entity against the topic's Wikipedia page text.
- We publish labels and scores, not page text. Wikipedia text is CC BY-SA.

## Pinned revisions
Recorded in `configs/` and in every run's `run_meta.json` (filled in at Step 2).
