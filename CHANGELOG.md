# Changelog

## 0.1.0
**Library**
- `Detector.from_pretrained("qwen3-4b")` with `generate()`, `score()`, `stream()` and `highlight()`; probe weights bundled.
- `hallustream` command: `ask` (default), `chat`, `score`, `batch`, with `--json` output.
- Risk shown as a calibrated percentage (calibration error 0.033 / 0.044 on held-out text); `balanced` and `strict` presets.
- Checks specific details (names, places, dates, numbers, titles, roles); refusals and pronouns are skipped.
- A SHARS plug-in (`hallustream.integrations.shars`).
- Measured with the shipped code: the same scores as the research code to within 0.0001, 0.876 / 0.827 AUROC with automatic entity finding, +5.0% time per generated token.
- Colab quickstart and a project page.

**Results** (details in [docs/RESULTS.md](docs/RESULTS.md))
- **Beats the paper it builds on, on that paper's own benchmark:** 0.894 / 0.897 AUROC vs 0.839 / 0.847 for its released Qwen2.5-7B probe, matching its LoRA fine-tuned probe without changing the model.
- **Inside SHARS (ICML 2026), on all 182 test people:** 44% fewer false claims than SHARS's own detector with the same true claims, higher precision, ~140× less detection time.
- **On the model's own writing:** 0.854 AUROC on 600 untouched people's biographies (the paper's recipe: 0.776) and 0.834 on 496 other subjects.
- **Fact checker** validated against FActScore's human labels (0.890 balanced accuracy on 11,049 facts).
- **Reproduced** Obeso et al.'s published probe results to within 0.0002 AUROC.
