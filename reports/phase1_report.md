# Phase 1 Report — ASAG Research Project (Data & Environment)

**Date:** 2026-05-30
**Owner:** Belal (belalasem19991@gmail.com)
**Scope:** Phase 1 only — reproducible env, dataset acquisition, EDA, validation, two-view preprocessing.
**Out of scope:** any modeling, training, or hyperparameter work — these belong to Phase 2.

---

## 1. Decisions and Rationale

We are building a hybrid explainable ASAG system: SBERT/DeBERTa semantic encoding + interpretable linguistic features + rubric-aware concept coverage → ordinal-regression head. The methodology is general but cross-domain evaluated, with **SAF Communication Networks English** chosen as the explainability case study because it ships gold answer-level feedback text.

### Dataset selection (verified 2026-05-29)

We acquired the report's full 5-dataset matrix (SemEval, ASAP-SAS, Mohler, SAF, Powergrading) and **added one new corpus**, **MIND-CA** (Kovatchev 2020, COLING), to widen cross-domain coverage with a non-STEM ordinal-graded dataset. See `reports/DATASETS.md` for the matrix + suitability ranking.

| Dataset | Role | Status |
|---|---|---|
| **SemEval-2013 Task 7** (Beetle + SciEntsBank, 5-way Core) | core + official UA/UQ/UD splits | ✅ downloaded |
| **SAF Communication Networks English** | explainability case study (feedback gold) | ✅ downloaded |
| **Mohler 2011** (canonical, extracted from ASAG2024) | ordinal grading + skew calibration | ✅ extracted |
| **Powergrading 1.0** (Basu 2013, MSR) | civics breadth, binary + 3-grader | ✅ downloaded |
| **MIND-CA** (Kovatchev 2020, COLING) — *added beyond the report* | true 3-class ordinal 0/1/2; new non-STEM domain (mindreading / psychology); large N | ✅ downloaded |
| **ASAG2024 unified benchmark** | row-level cross-check | ✅ downloaded |
| **ASAP-SAS** (Hewlett) — via the AERA mirror | rubric / QWK (Phase 2 head-to-head) | ✅ downloaded (science/biology subset, 4 of 10 prompts) |

Excluded: **EngSAF** (gated, request-only); **SAS-Bench** (Chinese only); **Carousel K-12** (not publicly released yet).

### ASAP-SAS source — free mirror instead of the gated Kaggle competition

The official ASAP-SAS data ("The Hewlett Foundation: Short Answer Scoring") is gated behind manual Kaggle competition-rule acceptance. To keep Phase 1 fully reproducible without a Kaggle account, we acquire it from the **AERA** mirror (Li et al., *Distilling ChatGPT for Explainable Automated Student Answer Assessment*, Findings of EMNLP 2023; CC-BY-NC-4.0, `jiazhengli/AERA`). The mirror republishes the **science/biology prompts (EssaySets 1, 2, 5, 6 — 4 of the 10 ASAP-SAS sets)** with both human-rater scores (`Score1`, `Score2`) and, unlike the original Kaggle release, **gold scores on the test split**, which we map to `test_ua`. It also ships an `llm_rationale` column per response, reserved for the Phase 2 explainability study. **Documented limitation:** this is a 4-prompt subset, not the full 10-prompt corpus; Phase 2 can substitute the full Kaggle data if a wider QWK comparison is required (the loader reads the identical `EssaySet`/`EssayText`/`Score1` columns, so no code change is needed).

### Mohler source — important discovery

The Kaggle dataset suggested in the original plan (`mubeenfurqanahmed/automatic-short-answer-grading-dataset`) is **NOT actual Mohler 2011** — it contains questions about plant respiration, meridians, etc., not CS data structures. We dropped that source and use the Mohler subset from ASAG2024 as the canonical source. The ASAG2024 Mohler subset is smaller than the original 2011 corpus (21 unique questions / 1,260 rows vs original ~80 questions / 2,273 rows) — documented as a Phase 1 limitation; we may re-acquire the original UMich corpus via a stable mirror in Phase 2 if larger N is required.

### Environment

- **Python 3.11** with **`uv`** for env + dep management.
- All deps pinned in `pyproject.toml`.
- spaCy model `en_core_web_sm`.
- Global seed `42`.
- sha256 manifest at `data/raw/CHECKSUMS.txt`.
- **Windows non-ASCII path workaround**: venv lives at `%USERPROFILE%/.cache/asag-venvs/asag-py311` because Python 3.11's `site.addpackage` reads `.pth` files via cp1252 and fails on Arabic-path bytes. `PYTHONUTF8=1` + `PYTHONIOENCODING=utf-8` exported throughout. Documented in `README.md`.

### Reproducibility

- `make setup` → clean venv + deps + spaCy model.
- `make download` → idempotent acquisition with sha256 verification.
- `make validate` → JSON reports per dataset under `reports/validation/`.
- `make preprocess` → two-view parquets under `data/processed/`.
- `make eda` → executes `notebooks/01_eda.ipynb` end-to-end.
- `make test` → pytest smoke tests.

---

## 2. Data Stack — Final Statistics

| Dataset | Domain | N_q | N_a (raw) | N_a (after dedup) | Mean tokens | Score range | Splits |
|---|---|---:|---:|---:|---:|---|---|
| **SemEval-2013 Task 7** | electronics + science | 252 | 16,003 | — | 11 | 5-way categorical | train, test_ua, test_uq, test_ud |
| **SAF Comm. Networks** | comm_networks | 31 | 2,981 | — | 69 | 0.0 – 3.5 | train, dev, test_ua, test_uq |
| **Mohler (via ASAG2024)** | cs_data_structures | 21 | 1,260 | 616 | 18 | 0.0 – 5.0 | all (k=5 CV) |
| **Powergrading** | civics | 20 | 13,960 | 4,941 | 3 | 0.0 / 0.5 / 1.0 (binary, 3-grader mean) | all (k=5 CV) |
| **MIND-CA** *(new domain)* | mindreading_behavioral (children 7–14) | 11 | 11,311 | 11,311 (no dups) | 10 | **0 / 1 / 2 ordinal** (skew −0.07, near-uniform) | all (k=5 CV) |
| **ASAP-SAS** (AERA mirror) | science + biology (4 prompts) | 4 | 8,722 | 8,722 (official splits, no dedup) | 37 | **0–3 ordinal** | train, dev, test_ua |

**Totals across acquired datasets: 339 unique questions, 44,574 answers (post-dedup where applicable), 7 distinct domains** (science, electronics, comm_networks, cs_data_structures, civics, mindreading_behavioral, biology).

(See `reports/figures/dataset_summary.png` for the rendered table and per-dataset distribution figures. SemEval/SAF use their official splits as-is; Mohler/Powergrading/MIND-CA are deduped by `preprocess.dedupe_within_question` keeping the median-score row per (question_id, student_answer) group. MIND-CA had zero exact duplicates — child answers are textually varied enough that even when scores match, the text differs.)

### Score / label distributions (`reports/figures/score_or_label_dist.png`)

- **SemEval (categorical 5-way)**: `correct` ~6.6k > `partially_correct_incomplete` ~3.8k > `irrelevant` ~2.8k > `contradictory` ~2.5k > `non_domain` ~0.3k. Class imbalance to address in Phase 2.
- **SAF (0.0–3.5)**: bimodal — clusters at 0.0, 0.5, 1.0, and a tail to 3.5; positive skew (1.65). Most answers are partially correct.
- **Mohler (0–5)**: heavy negative skew (-1.34) toward 5 — confirms the well-known Mohler-skew issue. Phase 2 ordinal-regression head must compensate via class-balanced sampling or weighted loss.

### Answer length (`reports/figures/answer_length.png`)

- SemEval p95 = 25 tokens — very short (single-sentence Q&A).
- SAF p95 = 165 tokens — longer multi-sentence explanations.
- Mohler p95 = 43 tokens — moderate.

Phase 2 will pick encoder max-length per dataset rather than using a single global cap.

### SAF label↔score sanity (`reports/figures/saf_score_by_label.png`)

Boxplot confirms monotonic alignment: `Incorrect` → 0; `Partially correct` → median ~0.6 (IQR ~0.5–1.0); `Correct` → median ~1.0. A handful of outliers exist where labels and scores disagree — small (<2% per group), acceptable noise.

---

## 3. Two-View Preprocessing

The pipeline produces two text views per row:

* **Encoder view** (SBERT/DeBERTa input): NFKC + whitespace normalization only. No lowercasing, no punctuation removal, no stopword removal, no stemming. Preserves everything the transformer was pre-trained on.
* **Feature view** (handcrafted linguistic features): spaCy lemmatization + lowercase + punctuation removal + stopword removal, with **negators preserved** (`not, no, never, n't, without, neither, nor, none, cannot, can't, won't, don't, doesn't, didn't, isn't, wasn't, weren't, shouldn't, wouldn't, couldn't`).

Both views are written per dataset to `data/processed/<dataset>/encoder.parquet` and `feature.parquet`, plus JSONL backups and a `_sidecar.json` describing row counts and one sample row.

---

## 4. Splits

* **SemEval / SAF**: official `train / dev / test_ua / test_uq / test_ud` splits recorded in the `split` column and used as-is. Never mixed.
* **SemEval Core only**: each split directory contains `Core/`, `Extra/`, and (sometimes) `Dependency/` subdirs — these are alternative annotation styles over the **same** student responses. Reading all of them doubles rows; standard practice in the ASAG literature is to use **Core**. We restrict to `Core/`.
* **ASAP-SAS** (AERA mirror): each prompt (`EssaySet`) is a distinct `question_id` (`set_1/2/5/6`); the mirror's `train`/`val`/`test` map to `train`/`dev`/`test_ua`. Because the same prompts recur across splits, `test_ua` is *unseen answers* (not unseen questions) — so prompt-id recurrence across train↔test_ua is expected and is **not** flagged as leakage (the structural `question_id` leakage check fires only on `test_uq`/`test_ud`). Scoring scales are never merged across prompts.
* **Mohler**: no official splits → stratified k=5 CV over binned scores (`asag.data.splits.make_stratified_kfold`); the fold index is materialized in the `fold` column of `data/processed/mohler/*.parquet`.

---

## 5. Validation Findings (`reports/validation/summary.json`)

| Dataset | n_rows | schema_ok | exact_dups | near_dups | leakage |
|---|---:|---|---:|---:|---|
| semeval | 16,003 | ✅ | 527 | 0 | qid: 0 across test_uq+test_ud ✅; answer-text overlaps with train: test_ua=86, test_ud=10, test_uq=105 (natural — very short answers like "yes"/"no") |
| saf | 2,981 | ✅ | 57 | 2 | qid: 0 across test_uq ✅; answer-text overlaps: dev=13, test_ua=8, test_uq=0 |
| mohler | 1,260 | ✅ | 644 | 0 | n/a (no official train split) |
| asap_sas | 8,722 | ✅ | 31 | 0 | qid: 0 across test_uq/test_ud ✅ (test split is test_ua by design); answer-text overlaps with train: dev=9, test_ua=20 (tiny identical short answers) |
| powergrading | 13,960 | ✅ | 9,019 | 0 | n/a (no official train split) |
| mindreading | 11,311 | ✅ | 0 | 0 | n/a (no official train split) |

**Interpretation:**
- **No question_id leakage in any unseen-question/unseen-domain split.** The cross-domain claim is structurally clean.
- Student-answer text overlaps in SemEval/SAF are tiny relative to dataset size and consist of common short responses — not a leakage problem.
- The Mohler exact-dup count (644 / 1,260 = 51%) is driven by ASAG2024's row-level handling: identical (question, provided_answer) tuples appear multiple times.
- The Powergrading exact-dup count (9,019 / 13,960 = 65%) is natural: 698 students answering 20 civics questions inevitably produces many identical short responses ("the Bill of Rights", "freedom of speech").
- **MIND-CA has zero exact duplicates** despite 11,311 child responses across only 11 prompts — children produce textually varied answers even when the underlying score matches, validating the choice of corpus for ordinal-head training.
- **Mohler, Powergrading, and MIND-CA are all deduped in `preprocess.py`** via `dedupe_within_question`, which keeps the median-score row per duplicate group (avoiding "best/worst answer" bias). This addresses the report's Section 2.3 reviewer hot-button for Mohler explicitly.
- **ASAP-SAS has only 31 exact duplicates** (0.4%) and zero near-duplicates — these are identical short answers across official splits and are intentionally **not** deduped, since doing so would corrupt the official train/dev/test partition. The `question_id` leakage check is clean: the only recurrence is prompt ids in `test_ua`, which is correct (unseen answers, seen prompts).

---

## 6. Risks (open items for Phase 2)

* **ASAG2024 Mohler size** — only 21 questions / 1,260 raw / ~616 deduped rows. If we need closer to original Mohler size (~80 q / 2,273 rows), Phase 2 should pursue a more authoritative Mohler mirror.
* **Mohler exact dups (51%)** — addressed: `preprocess.dedupe_within_question` drops them keeping median-score row.
* **Powergrading exact dups (65%)** — addressed: same dedup function.
* **SemEval class imbalance** — `non_domain` is ~2% of labels; will need class-balanced sampling or loss weighting in Phase 2.
* **Mohler score skew toward 5 (skew = -1.34)** — ordinal-regression head must compensate via weighted loss in Phase 2.
* **ASAP-SAS coverage** — acquired via the AERA mirror, but only 4 of the 10 prompts (science/biology). Sufficient for a QWK comparison in Phase 2; if a broader 10-prompt comparison is needed, substitute the full Kaggle competition data (loader is column-compatible, so no code change is required). The original gated source is no longer a blocker.
* **License diversity** — every dataset has its own terms; redistribution is forbidden; `data/raw/**` is gitignored; users acquire under their own license acceptance.
* **Windows non-ASCII path bug** — Python 3.11 fails to read `.pth` files via cp1252 on Arabic-path venvs; mitigated by relocating venv to ASCII path. Documented in `README.md`.

---

## 7. Next Steps — Phase 2 Preview

Phase 2 will:

1. Build the hybrid model: SBERT/DeBERTa encoder on the encoder view + linguistic + rubric features on the feature view + concept-coverage scoring against the reference answer + ordinal-regression head.
2. Train per-dataset and evaluate **cross-domain** using the official UA/UQ/UD splits prepared here.
3. Report Pearson / RMSE / QWK by split, plus per-feature attributions for the SAF explainability case study.
4. Use ASAP-SAS (already acquired, with both rater scores) for the headline QWK comparison; optionally widen to all 10 prompts via the full Kaggle release.
5. Address the Mohler-acquisition gap (re-acquire larger version via a stable mirror, or drop Mohler if the ASAG2024 subset turns out to be insufficient).
