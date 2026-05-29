# Phase 1 Report — ASAG Research Project (Data & Environment)

**Date:** 2026-05-29
**Owner:** Belal (belalasem19991@gmail.com)
**Scope:** Phase 1 only — reproducible env, dataset acquisition, EDA, validation, two-view preprocessing.
**Out of scope:** any modeling, training, or hyperparameter work — these belong to Phase 2.

---

## 1. Decisions and Rationale

We are building a hybrid explainable ASAG system: SBERT/DeBERTa semantic encoding + interpretable linguistic features + rubric-aware concept coverage → ordinal-regression head. The methodology is general but cross-domain evaluated, with **SAF Communication Networks English** chosen as the explainability case study because it ships gold answer-level feedback text.

### Dataset selection (verified 2026-05-29)

| Dataset | Role | Status |
|---|---|---|
| **SemEval-2013 Task 7** (Beetle + SciEntsBank, 5-way Core) | core + official UA/UQ/UD splits | ✅ downloaded |
| **SAF Communication Networks English** | explainability case study (feedback gold) | ✅ downloaded |
| **Mohler 2011** (canonical, extracted from ASAG2024) | ordinal grading + skew calibration | ✅ extracted |
| **ASAG2024 unified benchmark** | row-level cross-check | ✅ downloaded |
| **ASAP-SAS** (Hewlett) | rubric / QWK (stretch goal) | ⏸ disabled by default |

Excluded: **EngSAF** (gated, request-only).

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

| Dataset | Domain | N_q | N_a | Mean tokens | Score range | Splits |
|---|---|---:|---:|---:|---|---|
| **SemEval-2013 Task 7** | electronics + science | 252 | 16,003 | 11 | 5-way categorical | train, test_ua, test_uq, test_ud |
| **SAF Comm. Networks** | comm_networks | 31 | 2,981 | 69 | 0.0 – 3.5 | train, dev, test_ua, test_uq |
| **Mohler (via ASAG2024)** | cs_data_structures | 21 | 1,260 | 18 | 0.0 – 5.0 | all (k=5 CV) |

(See `reports/figures/dataset_summary.png` for the rendered table and per-dataset distribution figures.)

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
* **ASAP-SAS** (when enabled): each prompt (`EssaySet`) treated independently; scoring scales never merged across prompts.
* **Mohler**: no official splits → stratified k=5 CV over binned scores (`asag.data.splits.make_stratified_kfold`); the fold index is materialized in the `fold` column of `data/processed/mohler/*.parquet`.

---

## 5. Validation Findings (`reports/validation/summary.json`)

| Dataset | n_rows | schema_ok | exact_dups | near_dups | leakage |
|---|---:|---|---:|---:|---|
| semeval | 16,003 | ✅ | 527 | 0 | qid: 0 across test_uq+test_ud ✅; answer-text overlaps with train: test_ua=86, test_ud=10, test_uq=105 (natural — very short answers like "yes"/"no") |
| saf | 2,981 | ✅ | 57 | 2 | qid: 0 across test_uq ✅; answer-text overlaps: dev=13, test_ua=8, test_uq=0 |
| mohler | 1,260 | ✅ | 644 | 0 | n/a (no official train split) |

**Interpretation:**
- **No question_id leakage in any unseen-question/unseen-domain split.** The cross-domain claim is structurally clean.
- Student-answer text overlaps are tiny relative to dataset size and consist of common short responses — not a leakage problem.
- The Mohler exact-dup count (644 / 1,260 = 51%) is driven by ASAG2024's row-level handling: identical (question, provided_answer) tuples appear multiple times in the unified benchmark. We will dedupe Mohler before training in Phase 2 — drop or downweight.

---

## 6. Risks (open items for Phase 2)

* **ASAG2024 Mohler size** — only 21 questions / 1,260 rows. If we need closer to original Mohler size (~80 q / 2,273 rows), Phase 2 should pursue a more authoritative Mohler mirror.
* **Mohler exact dups (51%)** — must be removed or downweighted before training; affects sample-weight strategy.
* **SemEval class imbalance** — `non_domain` is ~2% of labels; will need class-balanced sampling or loss weighting.
* **Mohler score skew toward 5** — ordinal-regression head must compensate.
* **ASAP-SAS gating** — requires manual Kaggle setup (account + accept rules + `~/.kaggle/kaggle.json`); script ready, disabled by default in `configs/data.yaml`.
* **License diversity** — every dataset has its own terms; redistribution is forbidden; `data/raw/**` is gitignored; users acquire under their own license acceptance.
* **Windows non-ASCII path bug** — Python 3.11 fails to read `.pth` files via cp1252 on Arabic-path venvs; mitigated by relocating venv to ASCII path. Documented in `README.md`.

---

## 7. Next Steps — Phase 2 Preview

Phase 2 will:

1. Build the hybrid model: SBERT/DeBERTa encoder on the encoder view + linguistic + rubric features on the feature view + concept-coverage scoring against the reference answer + ordinal-regression head.
2. Train per-dataset and evaluate **cross-domain** using the official UA/UQ/UD splits prepared here.
3. Report Pearson / RMSE / QWK by split, plus per-feature attributions for the SAF explainability case study.
4. Optionally enable ASAP-SAS to strengthen QWK comparison once Kaggle credentials are configured.
5. Address the Mohler-acquisition gap (re-acquire larger version via a stable mirror, or drop Mohler if the ASAG2024 subset turns out to be insufficient).
