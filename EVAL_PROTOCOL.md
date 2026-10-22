# Evaluation protocol (pre-registered)

This file was committed **before** any probe was trained. Its purpose is to fix in advance how results are measured, so they can't be tuned after the fact. Any later change is logged in `CHANGELOG.md` with the reason.

## What is scored
- **Unit:** one labeled entity span (a name, date, number, place or citation), scored by the **highest** per-token risk inside it.
- **Labels:** "Not Supported" and "Insufficient Information" = positive (hallucinated); "Supported" = negative; missing labels are ignored.
- **Own-answer data (planned):** "Contradicted" = positive; "Supported" = negative; "Not on the page" is dropped.

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
- **Rule, fixed before training the final probe:** train the final recipe on both layer 17 (A) and layers 17 + 30 (B). Ship B only if, on validation:
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

## Own-answer gate: biographies (decided before any biography is generated)
- **Question:** does the frozen probe catch hallucinated entities in text the model wrote itself? So far it has only scored text written by other models.
- **Models:** Qwen3-4B-Instruct-2507 (layer 21 probe) is the pipeline model for the next steps. Qwen3.5-4B (layers 17 + 30) is reported alongside it.
- **Prompts:** the 500 people in FActScore's unlabeled set, as `Tell me a bio of {name}.` Thinking off, greedy decoding, at most 512 new tokens.
- **Labels:** an LLM judge (`gemini-3.1-flash-lite`, Batch API) reads the biography plus the person's Wikipedia page (`enwiki-20230401`, FActScore's passages): every passage, or the 24 best-matching ones (BM25 against the biography) if the page is longer. It labels every entity *Supported*, *Not Supported* or *Insufficient*, and every atomic claim *Supported* or *Not Supported*. The judge prompt is fixed in `src/hallustream/research/factcheck/prompts/bio_judge.txt` (labeling check: `claim_judge.txt`).
- **Primary analysis:** entity AUROC, *Not Supported* (positive) vs *Supported* (negative), with *Insufficient* dropped. The probe score is the mean of the 5 frozen seeds (each seed's AUROC is also reported). A 95% interval comes from 1,000 bootstrap resamples over people. The baseline is the one frozen for that model in `configs/frozen_choices.json` (Qwen3-4B: max entropy; Qwen3.5-4B: max surprise), scored on the same entities with its own interval. The pass rule uses the point estimate; the interval is reported.
- **Secondary analysis:** *Insufficient* counted as positive. The judge's *Insufficient* (Wikipedia is silent on a minor detail) is narrower than the training data's *Insufficient Information*, so this is reported, not used for the pass rule.
- **Also reported:** each model's response rate (biographies with no entities, mostly "I don't know this person"), AUROC on the people both models answered, the share of judged entities that could not be placed in the text, the share of biographies cut off at 512 tokens (entities touching the cut-off end are dropped) and AUROC without them, and how often teacher forcing reproduces the greedy tokens (a check that re-tokenizing the text did not change it).
- **People the probe has seen.** The probe's training data includes other models' biographies of 301 of the 500 people (`longfact_augmented_biography`, 445 of 20,056 training answers; list in `configs/bio_people_in_probe_data.json`, found before any gate result). The gate is also reported separately for people with no training answer. If their AUROC is more than 0.05 below that of the seen people, the unseen-people AUROC is the headline number and the one used for the pass rule.
- **The judge is checked first.** The 183 human-labeled people are split once at random (seed 0) into 40 tuning people and 143 reported people.
  - **Tuning (40 people):** four judge settings are compared, temperature 1.0 or 0 crossed with thinking level `low` or `minimal`, on the two checks below. The setting with the highest labeling balanced accuracy among those within 0.03 end-to-end error is chosen (ties: lower end-to-end error, then cheaper). The choice is recorded here before the reported run.
  - **Reported (143 people × 3 systems: InstructGPT, ChatGPT, PerplexityAI), both must pass:**
    - **Labeling:** given the humans' own atomic facts, the judge's labels reach balanced accuracy ≥ 0.80 against human S/NS labels (IR excluded). A biography whose answer is missing, malformed or skips a claim counts as failed and is reported.
    - **End to end:** with the judge's own claim splitting, each system's FActScore (supported claims over all claims, averaged over biographies) is within 0.03 of the human FActScore, computed as in the paper: supported facts over all facts, irrelevant (IR) facts counting as unsupported. Claims per biography (judge vs human) and failed biographies are reported.
  - **Entity check (reported):** on the same biographies, an entity mentioned only in facts the humans marked S should be *Supported*, and one mentioned only in NS facts *Not Supported* (balanced accuracy). About 100 gate entities are also checked by hand.

  - **Tuning result (recorded before the reported run):** chosen setting `t1_minimal` (temperature 1.0, thinking `minimal`). On the 40 tuning people: labeling balanced accuracy 0.873–0.877 for all four settings; entity check 0.91; end-to-end error 0.025 (InstructGPT), 0.022 (ChatGPT), 0.071 (PerplexityAI). The judge splits about half as many claims per biography as the humans (13–18 vs 25–39), and an extra instruction to split more finely did not change that.
  - **Amendment (made on the tuning people only):** the end-to-end bar applies to InstructGPT and ChatGPT, which write from memory like the models we test; PerplexityAI's error is reported, not used for the pass/fail decision. Reason: FActScore's annotators checked facts against all of English Wikipedia, while the judge sees only the person's page, and PerplexityAI copies web search results; in 7 of 8 sampled disagreements the fact was not on the person's page, and FActScore's own estimators miss PerplexityAI by up to about 9–10 points (their Table 3).

  If either check fails, the judge is upgraded (`gemini-3.5-flash-lite`, then a stronger model) and re-checked on the tuning people, then once on the reported people, before any gate result is read.
- **Pass rule:**
  - **AUROC ≥ 0.75 on Qwen3-4B:** proceed with detect→fix on Qwen3-4B.
  - **Otherwise:** retrain the probe with own-answer labels (people split into train/val/test) and re-test once.
  - **If it still fails:** the main direction becomes scoring other models' text (reader mode).

## Detector v2 on own answers (decided before any dev biography is generated)
- **Why:** the gate's probe (v1) was trained only on other models' text. On a model's own text, its confidence is a stronger signal (0.76 vs 0.63–0.68), so a detector that uses both, or a probe adapted to own answers, may do better. v1's gate result stays the pre-registered result; v2 is reported next to it.
- **Test set stays untouched:** the 500 gate people are not used for any v2 choice.
- **Dev people:** 1,200 Wikidata humans with an English Wikipedia page in `enwiki-20230401.db`, none of them among the 500 gate people, the 183 labeled people, or any LongFact biography person. They are sampled equally from 4 popularity buckets by Wikidata sitelink count (1–3, 4–10, 11–30, 31+). 600 (150 per bucket) are used here; the other 600 are reserved for later training. The list is committed as `configs/dev_people.json`.
- **Model:** Qwen3-4B only, with the same generation and judge settings as the gate (`t1_minimal`).
- **Judge audit on own text:** 300 dev entities (100 per judge label) are re-judged by `gemini-3.5-flash` with the person's whole page. Agreement on Supported vs Not Supported is reported; below 0.85, the judge is revisited using agreement only, never detector scores.
- **Candidates** (v1 = mean of the 5 frozen seeds, max over the entity's tokens):
  - **A. Combiner:** logistic regression on v1's logit, max and mean surprise, max and mean entropy, span length in tokens, and relative position in the biography.
  - **B. Adapted probe:** v1's logit plus a learned linear term on the entity's mean hidden state at one layer (15, 18, 21, 24, 27 or 30), with L2 strength chosen inside the folds.
  - **C. A + B.**
  - **D. Token aggregation:** v1's score as max, mean or last token over the entity, applied to the winning candidate.
- **Choice rule:** 5-fold cross-validation on the 600 dev people, folds by person. Per-candidate hyperparameters are tuned in inner folds. The gain over v1 on out-of-fold predictions gets a 95% interval from 2,000 bootstrap resamples over people. The simplest candidate (order A, D, B, C) with a gain ≥ 0.01 AUROC and an interval above 0 is chosen; if none qualifies, v1 is kept. The chosen detector is refit on all 600 dev people, frozen in `configs/detector_v2.json`, and then scored once on the 500 gate people.

## Detector v3 and a final check on untouched people (decided before any reserved biography is generated)
- **Why:** an audit found candidate B's penalty grid (hidden-state L2 up to 1.0) far too weak: every fold picked the grid's edge and the fit memorized the training folds. With stronger penalties, B is clearly better on dev, so "refitting on own answers did not help" was a grid artifact. Also, v2's `position` input uses the finished biography's length, so it is not available while the model writes.
- **Dev choice:** the same 432 dev people, the same nested 5-fold, person-grouped procedure and the same rule (simplest candidate, gain ≥ 0.01 with the interval above 0, now measured over v2) choose between v2 and new candidates. The new candidates are A without `position`, B and C with hidden-state penalties of 30 to 10,000 (per dimension) at layers 15–30, with C's tabular penalty at 0.001 or 0.01. The grid was widened after seeing that a stronger penalty helps, so dev numbers for B and C are slightly optimistic; the final check below is what counts. The winner is frozen in `configs/detector_v3.json`.
- **Final check:** the 600 reserved people in `configs/dev_people.json`, untouched so far: same generation, judge and scoring. v1, v2 and v3 are each scored once on them, with intervals from 2,000 bootstrap resamples over people. The gate people are not used again. The detector for detect→fix is v3 if it beats v2 there with the interval above 0; otherwise v2.

## Detect→fix on biographies (decided before any fix is generated)
- **Question:** does a probe-triggered fix (rewrite a flagged sentence with Wikipedia evidence) reduce hallucinations better than other triggers that use the same number of retrievals, and does a probe guard add value on top of retrieving once up front?
- **Model and people:** Qwen3-4B-Instruct-2507, greedy decoding, `Tell me a bio of {name}.`, at most 512 new tokens. FActScore's 500 people are split once at random (seed 0) into 100 dev people, used for every choice, and 400 test people, scored once.
- **Judge:** the validated judge (`gemini-3.1-flash-lite`, `t1_minimal`, person's Wikipedia page). Precision is supported claims over all claims.
- **Retrieval:** BM25 over the person's own Wikipedia page (`enwiki-20230401`), query = name + the sentence being fixed, top 3 passages (top 5 for retrieve-once), placed in the user turn after the question. This is the same source the judge uses (as in Self-RAG's and CtrlA's bio setups), so retrieval-vs-no-retrieval gains are inflated; comparisons between retrieval arms share this and stay fair.
- **Detector:** v3 on entity spans found while the model writes; the span method and the sentence trigger are chosen in the pilot.
- **Pilot (dev people only), with fixed decisions:**
  1. Entity spans: recall of GLiNER-medium, spaCy and probe-token runs against the judge's entities, especially Not Supported ones; the method with the highest Not Supported recall at ≤ 100 ms per sentence is used.
  2. Sentence trigger: AUROC for "the sentence contains a Not Supported entity" of v3 (max over the sentence's spans), the max token probe score, the minimum token probability (FLARE-style) and the max entropy. The best probe-based and the best uncertainty-based trigger go forward.
  3. Retrieval-augmented text: retrieve-once biographies are generated and judged. The probe guard on top of retrieve-once (arm A9) runs only if at least 5% of their entities are Not Supported and the probe's AUROC on them is at least 0.75 (scored with or without the evidence in context, whichever is higher on dev); otherwise it is reported as a negative result.
- **Dev tuning:** thresholds for each trigger so that it averages 1 and 2 fixes per biography; plus, for the probe trigger at 1 fix per biography, (a) evidence for the flagged sentence only vs kept for the rest of the biography, and (b) accepting the rewritten sentence vs re-scoring it, retrying once and dropping it if still flagged. The choices with the higher dev precision, without losing more than 10% of supported claims, are frozen here before the test run.
- **Test arms (400 people):** A0 greedy; A1 retrieve-once; probe trigger at 1 and 2 fixes per biography; uncertainty trigger at the same rates; random trigger at the probe's rate; oracle trigger (sentences the judge marked wrong in A0); probe trigger → drop the sentence; A9 retrieve-once + probe guard and A9 with the uncertainty trigger instead (if the pilot allows).
- **Metrics:** precision; supported and unsupported claims per biography; response rate; FActScore with the length penalty (γ = 10); fixes and extra tokens per biography; fix success (flagged entity now Supported) and damage (supported claims lost). Paired bootstrap over people (2,000 resamples).
- **Claims tested:** (1) at matched fixes per biography, the probe trigger has higher precision than the uncertainty and random triggers, with the interval above 0; (2) A9 has higher precision than A1 without losing more than 5% of supported claims.

## Gates (decided in advance)
| Gate | Pass condition | If it fails |
|---|---|---|
| Reproduction (paper's Qwen2.5-7B probe through our pipeline) | AUROC within ±0.02 of 0.8383 (LongFact) and 0.8467 (LongFact++) | say "applies", not "reproduces" |
| Leak removal | ≤ 25% of training answers removed | use the LongFact++ test set only |
| Main result | as above | try 2-layer or small-network fallbacks, then report as-is |
| Own-answer (biographies) | judge checks pass, then AUROC ≥ 0.75 on Qwen3-4B (see above) | retrain on own-answer labels once, else switch to reader mode |
| Guard | stops ≤ 10% of answers | tighten the `guard` threshold |
| Neutral prompt (pasted text) | AUROC drop ≤ 0.03 | make the question field required |

## Fixed prompts
Added here **before** each is first used:
- the neutral default prompt: `Write a detailed, factual answer.`
- the own-answer entity-extraction and judge prompts
- the P(True) prompt
