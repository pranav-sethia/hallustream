# Data, models and licenses

## Annotated data

| Dataset | Used for | Notes |
|---|---|---|
| [`obalcells/longfact-annotations`](https://huggingface.co/datasets/obalcells/longfact-annotations) | train / val / test | The test split is the paper's "LongFact" set |
| [`obalcells/longfact-augmented-annotations`](https://huggingface.co/datasets/obalcells/longfact-augmented-annotations) | train / val / test | The test split is the paper's "LongFact++" set |

Both come from Obeso et al., *Real-Time Detection of Hallucinated Entities in Long-Form Generation* ([arXiv 2509.03531](https://arxiv.org/abs/2509.03531)). Their entity labels were made by Claude 4 Sonnet with web search, which agreed with human labels 84% of the time in the paper's check. The datasets have no license tag, so this project never re-hosts them; the code downloads them from the source at a pinned revision.

Training uses the official train splits of both datasets across all five source models. After removing any answer whose question also appears in a test set, that leaves 20,056 answers for training and 2,395 for validation (7.7% removed). Exact matching can miss reworded questions, so `research/leak_check.py` compares every test question with every train and validation question by TF-IDF similarity. Five of the 1,986 test questions had a match at 0.9 or above. Reading them, all five ask about the same subject, so their 7 train and validation answers are dropped before training. Pairs below 0.9 are mostly sibling topics that share a prompt template.

The test set is the official test splits of both datasets, all five sources: 9,426 answers. The `obalcells/hallucination-heads*` datasets aren't used, because they re-export the same answers and mixing them in would leak test data into training.

These are the only large public datasets with entity-level hallucination labels on long answers that real models wrote without any source documents, which is exactly the setting HalluStream targets. I looked at and skipped RAGTruth (it labels faithfulness to a given document, not factual recall), FAVA's training data (errors are inserted synthetically), HaluEval, ANAH and FELM (answer- or sentence-level labels, or not English) and PsiloQA (short answers, mostly not English, labeled by GPT-4o against a single passage).

## Models

| Use | Model | License |
|---|---|---|
| The shipped detector | `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 |
| Head-to-head with the paper, and its reproduction | `Qwen/Qwen2.5-7B-Instruct` + `obalcells/hallucination-probes` (`qwen2_5_7b_linear`) | Apache-2.0 |
| Finding details in the library | `urchade/gliner_medium-v2.1` | Apache-2.0 |
| Tests | `trl-internal-testing/tiny-Qwen3ForCausalLM` | – |

## Data written for this project

The people come from FActScore (Min et al., 2023), including SHARS's 182 test people, plus random Wikipedia people sampled across popularity levels (`configs/dev_people.json`). The other subjects are 500 random Wikipedia pages that aren't people (`configs/topic_subjects.json`). Qwen3-4B-Instruct-2507 wrote the texts with greedy decoding; the SHARS runs use SHARS's own decoding.

Every detail is checked by Gemini 3.1 Flash-Lite through the Batch API against the subject's Wikipedia page, using FActScore's enwiki-20230401 dump and the 24 best-matching passages by BM25. Only labels and scores are published, never page text, since Wikipedia text is CC BY-SA.

## Pinned revisions

- `obalcells/longfact-annotations` @ `949fd2f71b3efd403350e38539340d9d9ceefc9d`
- `obalcells/longfact-augmented-annotations` @ `63174c7ea9111adb6f4c4ea61895af3e3740d997`

These are set in `src/hallustream/research/data.py`. Every run also records its git commit in `run_meta.json`.
