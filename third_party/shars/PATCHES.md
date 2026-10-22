# Vendored SHARS

Source: https://github.com/TreeLLi/hallucination-rejection-sampling (MIT, see LICENSE), commit b7fcf64.
Paper: Li, Channing, Bhat, Davis Jones, Gal. "Building Reliable Long-Form Generation via
Hallucination Rejection Sampling" (ICML 2026).

Files copied unchanged except:
- `generator.py`: `Suppress.NOT` -> `SuppressEOS.NOT` (a NameError that the generator's broad
  `except` turned into a silently truncated answer).

Not copied: `main.py` (its loop stops after the first person and it requires wandb; our driver
`modal_jobs/shars.py` calls the same classes), `third_eval/` (we score with our own judge).
`factscore_people.txt` is their `data/factscore.txt` (182 people).
