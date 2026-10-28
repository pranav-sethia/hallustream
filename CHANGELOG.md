# Changelog

## 0.1.0

First release.

- `Detector.from_pretrained("qwen3-4b")` with `generate()`, `score()`, `stream()` and `highlight()`, with the probe weights bundled.
- A `hallustream` command with `ask` (the default), `chat`, `score` and `batch`, plus `--json` output.
- Risk is reported as a calibrated percentage, with `balanced` and `strict` presets.
- Only specific details are checked (names, places, dates, numbers, titles, roles). Refusals and pronouns are skipped.
- A plug-in that makes HalluStream the detector inside SHARS (`hallustream.integrations.shars`).
- A Colab quickstart and a project page.

Measured with this release, all on held-out data (details in [docs/RESULTS.md](docs/RESULTS.md) and [docs/LIBRARY.md](docs/LIBRARY.md)):

- On the source paper's own benchmark, 0.894 / 0.897 AUROC against 0.839 / 0.847 for its released Qwen2.5-7B probe, matching its LoRA fine-tuned probe without changing the model.
- Inside SHARS on all 182 test people, 44% fewer false claims than SHARS's own detector with the same true claims, higher precision, and about 140x less detection time.
- 0.854 AUROC on Qwen3-4B's biographies of 600 new people (0.776 for the paper's recipe) and 0.834 on 496 other subjects.
- The same scores as the research code to within 0.0001, and 5% extra time per generated token.
