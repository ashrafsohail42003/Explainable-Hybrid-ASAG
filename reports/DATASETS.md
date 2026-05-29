# Phase 1 — Dataset Cards

All links and licenses **independently verified on 2026-05-29** via WebFetch / WebSearch. Re-verify before publication.

---

## 1. SemEval-2013 Task 7 (Beetle + SciEntsBank)

| | |
|---|---|
| **Source** | https://github.com/myrosia/semeval-2013-task7 |
| **Files** | `semeval-3way.zip`, `semeval-5way.zip` |
| **License** | CC-BY-SA |
| **Subjects** | Beetle: basic electricity & electronics tutoring (DeepTutor). SciEntsBank: K–12 science (Nielsen et al., 2008). |
| **Size** | Beetle ~3k items; SciEntsBank ~10k items. |
| **Splits** | **train / test-unseen-answers (UA) / test-unseen-questions (UQ) / test-unseen-domains (UD)**. SciEntsBank has all four; Beetle has UA and UQ only. The UD split is the canonical cross-domain generalization test. |
| **Label scales** | 3-way (correct / contradictory / incorrect) and 5-way (correct / partially_correct_incomplete / contradictory / irrelevant / non_domain). |
| **Citation** | Dzikovska et al. (2013). *SemEval-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge.* |

**Why it's in our stack**: canonical ASAG benchmark; official UA/UQ/UD splits are essential to cross-domain evaluation and cannot be reconstructed from other sources.

---

## 2. SAF Communication Networks English

| | |
|---|---|
| **Source** | https://huggingface.co/datasets/Short-Answer-Feedback/saf_communication_networks_english |
| **License** | CC-BY-4.0 |
| **Subject** | College-level communication networks (telecommunications case study) |
| **Size** | **2,981 examples** over **31 questions** |
| **Splits** | train 1,700 · validation 427 · test_unseen_answers 375 · test_unseen_questions 479 |
| **Fields** | `id`, `question`, `reference_answer`, `provided_answer`, `answer_feedback`, `verification_feedback` (Correct / Partially correct / Incorrect), `score` (float, mostly 0–3.5) |
| **Citation** | Filighera, Parihar, Steuer, Meuser, & Ochs (2022). *Your Answer is Incorrect... Would you like to know why? Introducing a Bilingual Short Answer Feedback Dataset.* ACL 2022. |

**Why it's in our stack**: gold **answer-level feedback text** — the spine of our explainability case study — plus official UA/UQ splits.

---

## 3. Mohler 2011 (CS Data Structures)

| | |
|---|---|
| **Primary source** | Kaggle mirror: https://www.kaggle.com/datasets/mubeenfurqanahmed/automatic-short-answer-grading-dataset |
| **Original URL** | http://web.eecs.umich.edu/~mihalcea/downloads/ShortAnswerGrading_v2.0.zip — **❌ SSL verification fails on 2026-05-29**; mirror used instead. |
| **Cross-check** | Mohler subset extracted from https://huggingface.co/datasets/Meyerger/ASAG2024 (filter `data_source == "mohler"`). |
| **License** | Academic research use (original publication terms). |
| **Subject** | Introductory CS / data structures (10 assignments + 2 exams; 31 students). |
| **Size** | 2,273 answers / ~80 questions. |
| **Label scale** | 0–5 ordinal (average of two human graders). |
| **Splits** | No official splits → we use stratified k=5 CV (see `splits.py`). |
| **Known issue** | Heavy skew toward score 5; documented in EDA. |
| **Citation** | Mohler, Bunescu, & Mihalcea (2011). *Learning to Grade Short Answer Questions using Semantic Similarity Measures and Dependency Graph Alignments.* ACL 2011. |

**Why it's in our stack**: the canonical ordinal-grading ASAG dataset; calibrates the ordinal-regression head.

---

## 4. ASAP-SAS (Hewlett Foundation)  — stretch goal

| | |
|---|---|
| **Source** | https://www.kaggle.com/competitions/asap-sas |
| **License** | Kaggle competition terms (research use; redistribution restricted) |
| **Access** | Gated — requires (1) Kaggle account, (2) accepted competition rules, (3) `~/.kaggle/kaggle.json` credentials. |
| **Subjects** | 10 prompts across diverse subjects (biology, English, etc.). |
| **Size** | ~2,200 answers per prompt; ~22k total. |
| **Label scale** | 0–3 ordinal per prompt; **scales are NOT comparable across prompts** — each prompt is treated independently. |
| **Annotators** | Two graders per item (Score1, Score2); QWK between them is the IAA used in the literature. |
| **Citation** | Hewlett Foundation (2012). *Short Answer Scoring.* Kaggle. |

**Why it's in our stack (optional)**: provides rubric-aware ordinal data with dual annotations → QWK reporting. Enabled via `datasets.asap_sas.enabled: true` in `configs/data.yaml` after credential setup.

---

## 5. ASAG2024 (Meyerger / SIGCSE 2024) — cross-check only

| | |
|---|---|
| **Source** | https://huggingface.co/datasets/Meyerger/ASAG2024 · paper https://arxiv.org/abs/2409.18596 · code https://github.com/GeroVanMi/ASAG2024 |
| **License** | "Data source licenses apply" — each component dataset's license still applies. |
| **Composition** | Seven datasets unified: Beetle, CU-NLP, DigiKlausur, Mohler, SAF English, SciEntsBank, Stita. |
| **Size** | 56,646 rows; train 45,200 / val 5,700 / test 5,750 (**single random split — NOT official UA/UQ/UD**). |
| **Fields** | `question`, `provided_answer`, `reference_answer`, `grade` (0–100 raw), `normalized_grade` (0–1), `data_source`, `weight`, `index`. |
| **Citation** | Meyer, Breuer, & Fürst (2024). *ASAG2024: A Combined Benchmark for Short Answer Grading.* SIGCSE Virtual 2024. |

**Why it's in our stack**: used **only** to sanity-check our individually-downloaded datasets (row counts, content matches). We do NOT consume its splits because they discard the canonical UA/UQ/UD partitions our methodology depends on.

---

## Datasets considered but excluded from Phase 1

| Dataset | Reason excluded |
|---|---|
| **EngSAF** (2024) | Gated by request-only form; conflicts with the free-availability requirement. |
| **CU-NLP**, **DigiKlausur**, **Stita** | Available indirectly via ASAG2024; not pulled individually in Phase 1 to keep the primary stack focused. Can be added later if cross-domain breadth is needed. |
| **Powergrading** | Civic test answers; scope misalignment (no reference answer in our intended form). |

---

## License & redistribution summary

- All datasets are **research-use compatible**.
- **Commercial use** varies — at minimum: ASAP-SAS forbids redistribution; Mohler is academic-only; SAF/SemEval are CC-licensed.
- We **do not redistribute** any raw data via git; `data/raw/**` is gitignored. Users download under their own license acceptance.
