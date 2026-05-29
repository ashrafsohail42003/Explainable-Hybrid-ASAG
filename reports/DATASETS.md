# Phase 1 — Dataset Cards

All links and licenses **independently verified on 2026-05-29** via WebFetch / WebSearch. Re-verify before publication.

---

## Identity & Annotation Matrix (Section 2.1, extended)

| Dataset | Domain | Q / Answers | Label | Ref. ans. | Rubric | Partial credit | Verified access |
|---|---|---|---|---|---|---|---|
| **SemEval-2013 T7** (Beetle + SciEntsBank) | Science / electronics tutoring | 252 q (Core) / 16,003 a | 5-way categorical | ✅ | concept entailment | ✅ (categorical) | GitHub zips, HF mirror |
| **ASAP-SAS** | 10 mixed prompts (biology / English / civics / science) | 10 / ~17k | Ordinal 0–2 / 0–3 (per prompt) | rubric-based | explicit per prompt | ✅ (ordinal) | Kaggle (gated) |
| **Mohler 2011** | Computer Science (data structures) | 21 q / 1,260 a *(via ASAG2024 subset)* | Regression 0–5 | ✅ | ❌ | ✅ (continuous) | ASAG2024 HF (canonical Kaggle mirror rejected) |
| **SAF** (Filighera 2022) | Communication networks (EN) | 31 q / 2,981 a | Score 0.0–3.5 + gold feedback | ✅ | gold feedback text | ✅ | HF, GitHub |
| **Powergrading** (Basu 2013) | US citizenship / civics | 20 q / ~13,960 a | Binary (3 graders, majority) | ✅ (+ alternates) | ❌ | ❌ | Microsoft Download Center |
| **MIND-CA** (Kovatchev 2020, COLING) — *new domain* | Mindreading / behavioural psychology (children 7–14) | 11 q / 11,311 a | Ordinal 0/1/2 (true 3-class) | ❌ (copyrighted test materials) | ❌ | ✅ (ordinal) | GitHub raw |

## Suitability Matrix for THIS project (H/M/L) (Section 2.2, extended)

| Dataset | Ordinal reg. | Explainability | Semantic sim. | Rubric-aware | Cross-domain | Ablations | Popularity / Citations | Leakage risk | Q2/Q3 |
|---|---|---|---|---|---|---|---|---|---|
| **SemEval-2013** | M | M | H | M | **H** | H | H / H | split misuse | **H** |
| **ASAP-SAS** | **H** | M | M | **H** | L | H | H / H | per-prompt | **H** |
| **Mohler 2011** | **H** | M | H | L | L | H | H / H | skew + dup | M |
| **SAF** | M | **H** | H | **H** | L | H | L / M | small test | **H** |
| **Powergrading** | L | L | M | L | L | M | M / M | low | L |
| **MIND-CA** *(new)* | **H** | L (no rubric) | M | L | **H** *(new domain)* | H | M / M | low (per-task) | M |

**Ranking (best → worst for this idea):** SemEval-2013 > ASAP-SAS > Mohler 2011 > SAF > Powergrading.
**Best combination:** SemEval (all 3 splits) + ASAP-SAS + Mohler 2011 + SAF for the explainability study.
**Most publishable setup:** SemEval (UA/UQ/UD) + ASAP-SAS QWK head-to-head.
**Most novel:** SAF explainability evaluation (attributions vs gold feedback) + ASAG2024 cross-corpus generalization test.

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
| **Primary source (Phase 1)** | Mohler subset extracted from https://huggingface.co/datasets/Meyerger/ASAG2024 (filter `data_source == "mohler"`); written to `data/raw/mohler-2011/mohler_canonical_from_asag2024.parquet`. |
| **Original URL** | http://web.eecs.umich.edu/~mihalcea/downloads/ShortAnswerGrading_v2.0.zip — **❌ SSL verification fails on 2026-05-29**; not used. |
| **Kaggle "mirror" reviewed and rejected** | `mubeenfurqanahmed/automatic-short-answer-grading-dataset` — content inspection (2026-05-29) showed questions about plant respiration, meridians, evaporation — **NOT** Mohler 2011 CS data-structures. Excluded from the pipeline. |
| **License** | Academic research use (original publication terms). |
| **Subject** | Introductory CS / data structures (10 assignments + 2 exams; 31 students). |
| **Original size (literature)** | 2,273 answers / ~80 questions. |
| **Our extracted size** | **1,260 answers / 21 unique questions** — ASAG2024's Mohler subset is smaller than the original. Recorded as a Phase 1 limitation; we may pursue a larger mirror in Phase 2. |
| **Label scale** | 0–5 ordinal (ASAG2024 stores it on a 0–100 grade column; we keep raw). |
| **Splits** | No official splits → we use stratified k=5 CV (see `splits.py`). |
| **Known issues** | Heavy skew toward score 5 (skew = -1.34); 644 / 1,260 (51%) exact duplicates from ASAG2024 merging — addressed in Phase 2 via dedup/downweighting. |
| **Citation** | Mohler, Bunescu, & Mihalcea (2011). *Learning to Grade Short Answer Questions using Semantic Similarity Measures and Dependency Graph Alignments.* ACL 2011. |

**Why it's in our stack**: the canonical ordinal-grading ASAG dataset; calibrates the ordinal-regression head even at reduced size.

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

**Why it's in our stack (optional)**: provides rubric-aware ordinal data with dual annotations → QWK reporting. The report ranks ASAP-SAS as part of the "most publishable" pairing (alongside SemEval); we therefore plan its enablement for Phase 2 evaluation, gated on the user accepting competition rules.

**To enable in Phase 2** (one-time setup, ~5 minutes):
1. Sign in at https://www.kaggle.com/ and accept rules at https://www.kaggle.com/competitions/asap-sas/rules
2. Download API token from https://www.kaggle.com/settings/account → "Create New API Token"
3. Place the downloaded file at `%USERPROFILE%\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (POSIX). Restrict permissions: `chmod 600 ~/.kaggle/kaggle.json`.
4. Set `datasets.asap_sas.enabled: true` in `configs/data.yaml`.
5. Re-run `make download`. `download_asap_sas` will pull the competition zip into `data/raw/asap-sas/` and extract.
6. `load_asap_sas` materializes one row per (EssaySet, EssayText) pair with `dataset = "asap_sas_<id>"`; downstream evaluation MUST treat each prompt independently (no scale pooling).

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

## 7. MIND-CA (Kovatchev 2020, COLING)

| | |
|---|---|
| **Source** | https://github.com/venelink/mindreading-coling (Data/raw/*.xlsx) |
| **License** | CC-BY-4.0 |
| **Subject** | Mindreading / theory-of-mind / behavioural psychology assessments for children. |
| **Tasks** | 11 prompts: 5 *Strange Stories* (Brian, Burglar, Peabody, Prisoner, Simon) + 6 *Silent Films* questions. |
| **Population** | 1,066 children aged 7–14 (Age, Gender, Child_ID fields retained). |
| **Size** | **11,311 child responses**. |
| **Label scale** | True 3-class **ordinal 0 / 1 / 2** (poor / partial / full mindreading response). |
| **Reference answer / rubric** | ❌ Not provided. The expected-answer rubrics live in copyrighted psychology test materials (Happé's *Strange Stories* and *Silent Films*); the corpus releases only student answers + scores. We populate `reference_answer = ""` and use the task name as a synthetic prompt. |
| **Splits** | No official splits → stratified k=5 CV. |
| **Citation** | Kovatchev, Smith, Lee, Grumley Traynor, Luque Aguilera, & Devine (2020). *"What is on your mind?" Automated Scoring of Mindreading in Childhood and Early Adolescence.* COLING 2020, pp. 6217–6228. |

**Why included** (new domain, **not in the original report's 5-dataset matrix**):
- Adds a sixth genuinely cross-domain corpus (psychology / theory-of-mind), strengthening the cross-domain generalization claim with a non-STEM, non-civics target.
- True 3-class ordinal labels — directly calibrates the CORAL/CORN ordinal-regression head without the score-skew issue Mohler has.
- 11,311 responses from 1,066 children — a substantial sample size that complements our smaller science / CS corpora.
- The empty reference-answer column turns MIND-CA into a useful **ablation**: it forces the model to make decisions from `(question, student_answer)` only, isolating the semantic encoder's contribution from the rubric-coverage branch.

---

## 6. Powergrading 1.0 (Basu 2013)

| | |
|---|---|
| **Source** | https://www.microsoft.com/en-us/download/details.aspx?id=52397 (direct: https://download.microsoft.com/download/e/1/d/e1da2458-1af3-41c9-9515-7c9a8697e0cd/Powergrading-1.0-Corpus.zip) |
| **License** | MSR License Agreement for Powergrading-1.0 Corpus (research use) |
| **Subject** | US citizenship test (civics) — 20 questions from the USCIS 100-question list. |
| **Size** | 698 students × 20 questions = ~13,960 graded answers (plus 100 ungraded; we skip those). |
| **Annotators** | Three graders per response (G1, G2, G3); we report mean (mapped to 0/0.5/1) as `score` and majority vote as `label`. |
| **Splits** | No official splits → stratified k=5 CV (in preprocessing). |
| **Citation** | Basu, S., Jacobs, C., & Vanderwende, L. (2013). *Powergrading: a Clustering Approach to Amplify Human Effort for Short Answer Grading.* TACL 2013. |

**Why included** (note: ranked #5 in the suitability matrix): adds a clean **binary / triple-annotator** civics corpus for cross-domain breadth. Limited ordinal-grading signal (binary only); kept for completeness with the report's full 5-dataset matrix.

---

## Datasets considered but excluded from Phase 1

| Dataset | Reason excluded |
|---|---|
| **EngSAF** (2024) | Gated by request-only form; conflicts with the free-availability requirement. |
| **CU-NLP**, **DigiKlausur**, **Stita** | Available indirectly via ASAG2024; not pulled individually in Phase 1 to keep the primary stack focused. Can be added later if cross-domain breadth is needed. |

---

## License & redistribution summary

- All datasets are **research-use compatible**.
- **Commercial use** varies — at minimum: ASAP-SAS forbids redistribution; Mohler is academic-only; SAF/SemEval are CC-licensed.
- We **do not redistribute** any raw data via git; `data/raw/**` is gitignored. Users download under their own license acceptance.
