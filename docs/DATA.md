# Data, models, and licenses

## Annotated data (downloaded at runtime; never re-hosted)
| Dataset | Used for | Notes |
|---|---|---|
| [`obalcells/longfact-annotations`](https://huggingface.co/datasets/obalcells/longfact-annotations) | train / val / test | Test split = the paper's "LongFact" set |
| [`obalcells/longfact-augmented-annotations`](https://huggingface.co/datasets/obalcells/longfact-augmented-annotations) | train / val / test | Test split = the paper's "LongFact++" set |

Both datasets come from Obeso et al., *Real-Time Detection of Hallucinated Entities in Long-Form Generation* ([arXiv 2509.03531](https://arxiv.org/abs/2509.03531)). Their entity labels were produced by Claude 4 Sonnet with web search (84% agreement with human labels in the paper's check).

The datasets carry **no license tag**, so this project never redistributes them. Code downloads them from the source at a pinned revision.

## Which answers are used for training
- **Training and released probes use only answers written by Apache-2.0 or unrestricted-output models:** Qwen2.5-7B-Instruct, Mistral-Small-24B-Instruct-2501, Gemma-2-9B-it.
- **Llama-written answers are used for evaluation only.** The Llama 3.1/3.3 license (§1.b.i) requires any distributed AI model trained on Llama outputs to include "Llama" at the start of its name.

## Models
| Use | Model | License |
|---|---|---|
| Main | `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 |
| Reproduction check | `Qwen/Qwen2.5-7B-Instruct` + `obalcells/hallucination-probes` (`qwen2_5_7b_linear`) | Apache-2.0 |
| Tests | `trl-internal-testing/tiny-Qwen3ForCausalLM` | — |

## Own-answer data (Step 7)
- Qwen3 answers questions about Wikipedia topics.
- Claude Haiku 5.5 labels each entity against the topic's Wikipedia page text.
- We publish labels and scores, not page text. Wikipedia text is CC BY-SA.

## Pinned revisions
Recorded in `configs/` and in every run's `run_meta.json` (filled in at Step 2).
