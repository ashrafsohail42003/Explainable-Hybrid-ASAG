# Phase 1 — Project Understanding (reverse engineering) & Phase 2 — Research Contribution

All numbers below are **verified** from `reports/**` JSONs or source code unless labeled *(inference)* or *(missing)*.

---

## 1. What this project is

**Objective.** Grade free-text short answers (1–5 lines, English) against a question and usually a reference answer, producing a score/label **plus a teacher-auditable explanation**, and — centrally — measure this **honestly**, i.e., on questions the model has never seen.

**NLP task.** Automatic Short Answer Grading (ASAG): a hybrid of semantic textual similarity, textual entailment (SemEval 5-way labels), ordinal regression (ASAP-SAS, MIND-CA), and bounded regression (Mohler, SAF).

**Target users.** Instructors/TAs (per-answer concept-coverage explanations), assessment researchers (evaluation-protocol audit), system builders (deferral via risk–coverage).

**Paper-type verdict:** this is an **empirical / methodology paper** (evaluation protocol + validated explainability), *not* a model-novelty paper. The model (LightGBM over 31 features) is deliberately simple; the contribution lives in how it is evaluated and explained.

## 2. Pipeline (verified, module-by-module)

```
raw downloads (6 datasets, sha256, licenses logged)
  └─ loaders.py       → unified schema: question_id, question, reference_answer,
                        student_answer, score, label, dataset, domain, split
  └─ validate.py      → leakage checks (no q_id overlap train↔test_uq/ud), dup stats
  └─ preprocess.py    → TWO VIEWS per row:
       *_enc  : NFKC + whitespace only            → transformers/SBERT
       *_feat : spaCy lemma+lower+stop-drop, negators kept
       *_feat_neg : + neg_ prefix inside negation scope (window=4, clause-bounded)
       dedupe_within_question (median-score row) for no-official-split datasets
  └─ splits.py        → official splits kept; else GROUPED leave-questions-out k=5
                        (StratifiedGroupKFold on question_id, score-binned)
  └─ token_stats.py   → max_len=128 justified empirically (SAF outlier p99≈438 → 256)
  └─ features/        → 31 interpretable features, 3 branches:
       A sem_ (3): SBERT all-MiniLM-L6-v2 cosine, |u−v| mean, u⊙v mean
       B lex_/len_/tfidf_/neg_/ner_ (23): overlap, dice, n-grams, lengths,
         TF-IDF cosine, negation cues/mismatch, NER overlap
       C rub_ (5): per-reference-sentence concept coverage (tau=0.5),
         mean/min/max max-sim, n_concepts, coverage@tau
       (NaN by design for asap_sas & mindreading: 24/31 reference-dependent)
  └─ models/fusion.py → NaN-native LightGBM (clf / regression / ordinal=round+clip)
  └─ models/hpo.py    → Optuna TPE, 40 trials, dev-split or inner grouped CV only
  └─ models/evaluate.py → 3 arms: GBM, trivial baseline, QUESTION-SHORTCUT control;
                        + stratified "seen-question upper bound" + generalization_gap
  └─ models/significance.py → cluster bootstrap (resample QUESTIONS, 10k, one-sided
                        p(Δ≤0)) + Holm–Bonferroni across 6 datasets
  └─ models/ceiling.py → ASAP-SAS IAA: QWK(Score1,Score2) macro 0.9419 (train+dev)
  └─ xai/             → TreeSHAP (exact, native), faithfulness metrics,
                        rubric concept attribution, SAF gold-feedback validation
  └─ models/{ablations, leakage_audit, lodo, robustness, error_analysis,
             collapse_analysis, published_comparison}.py → Phase 3/4 reports
  └─ neural/          → DeBERTa-v3-base cross-encoder → OOF neural_score/neural_pred
                        → neural_oof.parquet → auto-fused as branch D (NOT YET RUN)
```

## 3. Hidden assumptions (found in code)

| # | Assumption | Where | Risk |
|---|---|---|---|
| 1 | Reference sentences ≙ rubric concepts | rubric.py | Proxy, not true rubric; degenerates for 1-sentence references |
| 2 | Negation scope = 4-token window, clause-bounded | config NegationScopeCfg | Misses wide scopes; window not ablated |
| 3 | Question difficulty ≙ per-question train mean | evaluate.question_shortcut | Fine as control; not a claim |
| 4 | LODO binary threshold = score-range midpoint | lodo.py | Arbitrary for ordinal scales; drives ASAP anti-transfer |
| 5 | Perturbations recompute only lexical+negation features | robustness.py | Conservative bound; semantic/rubric held fixed |
| 6 | Significance uses single seed 42 predictions | significance.py cfg.seed | Point estimates differ slightly from 5-seed means (e.g., SemEval 0.4331 vs 0.4269±0.0051) |
| 7 | English only (spaCy en_core_web_sm, English negators) | text_utils | Scope restriction, must be stated |
| 8 | SBERT embeddings cached; tau=0.5 fixed, untuned | semantic.py, config | tau untuned = honest; say so |

## 4. Limitations of the artifact (verified)

- **Subset corpora:** Mohler via ASAG2024 (21 q / 1,260 rows → 616 after dedupe; original: 80 q / 2,273). ASAP-SAS via AERA mirror (EssaySets 1,2,5,6 only). → no head-to-head claims vs published numbers (already documented in `published_comparison.md`).
- **Neural arm implemented, never executed** (no `neural_oof.parquet`; needs Colab GPU).
- **SAF unseen-question collapse:** test_ua Pearson 0.904 vs test_uq 0.019±0.033 — the features generalize almost not at all to new SAF questions; SAF is therefore an explainability case study, not an accuracy result.
- **MIND-CA near-floor:** QWK 0.077±0.003 unseen-question (no reference answer; only branch B active).
- Modest dataset sizes; no human study of explanation usefulness; single language.

## 5. Phase 2 — Contribution analysis

### What is genuinely there (verified, ranked by strength)

1. **Question-leakage audit across standard ASAG benchmarks** — quantified stratified-vs-grouped inflation: Powergrading macro-F1 0.8295→0.4500 (−0.38), Mohler Pearson 0.5698→0.4297 (−0.14), MIND-CA QWK 0.1206→0.0594 (−0.06); plus the *question-shortcut control* (question identity alone scores 0.8733 macro-F1 on Powergrading under stratified CV — nearly the full model's 0.8295) and per-dataset between-question η² (Powergrading 0.915, SAF 0.702). **Closest prior work:** Condor et al. (EDM 2021) showed the unseen-question drop for SBERT-ASAG; *nobody* (per July 2026 search) has published the audit-with-shortcut-control across Mohler/Powergrading/MIND-CA. → **primary novelty, honest and defensible.**
2. **Explanation validation against pre-existing human gold feedback (SAF)** — 4 interpretable signals all rise monotonically Incorrect→Partial→Correct (n=854; sem_cosine ρ=0.255, AUC=0.621). ExASAG (BEA 2023) validated SHAP against experts they recruited; using SAF's shipped `verification_feedback` as a free gold standard is a **novel mechanism** (scope the claim as "validation against pre-existing gold feedback").
3. **Faithfulness-vs-usefulness decomposition** — rubric branch is *faithful* (predictive-validity ρ 0.335, coverage signals track grades) yet adds ≈0 accuracy (ablation −C: −0.003 SemEval, +0.008 Mohler). An explanation layer that costs no accuracy is a quotable, reviewer-friendly finding.
4. **Cluster-bootstrap + Holm significance under the honest protocol** — reverses two apparent wins (Powergrading p=0.291, SAF p=0.582). Statistical hygiene rare in ASAG papers.
5. **Heterogeneous-benchmark engineering** — 6 datasets, 3 task types, unified schema, NaN-native fusion, full reproducibility (73 tests). Solid but not novel per se (ASAG2024 already unifies 7 datasets — must cite).
6. **LODO transfer with shared interpretable features** — weak-to-inverted transfer (ASAP AUC 0.269!) is an honest negative result; distinct from LLM-based cross-dataset work (S-GRADES 2026).
7. **(Pending) Hybrid neural-as-feature** — OOF DeBERTa signals fused as GBM features preserving TreeSHAP. GradeAid (KAIS 2023) already fuses BERT-similarity + TF-IDF → ours is incremental *unless* framed as: identical grouped-fold OOF protocol + explanation preservation + leakage-aware comparison of neural-only/feature-only/hybrid. That framing is publishable.

### Weak / missing novelty (be honest)

- The grouped split itself is standard ML practice (StratifiedGroupKFold) — the novelty is the **audit**, not the splitter.
- Feature set is conventional; SBERT bi-encoder is 2019-era. The model will not impress; the protocol must.
- No LLM baseline: in 2026 reviewers **will** ask "what does GPT-4/an open LLM score under your protocol?" — the strongest single addition possible (see Phase 5 doc).

### Classification & positioning

| Paper type | Fit |
|---|---|
| Methodology/empirical (evaluation protocol + validated XAI) | **primary — write this** |
| Benchmark/resource (unified splits + audit release) | secondary framing, mention code/splits release |
| Application/engineering | avoid — weakest version of this work |

**One-sentence contribution statement (for the paper):**
> We present a leakage-aware evaluation framework for ASAG that quantifies how much reported performance is question memorization, an interpretable fusion grader whose explanations are validated against human gold feedback, and a cluster-bootstrap significance analysis that reverses two apparent wins — with a DeBERTa hybrid arm evaluated under identical unseen-question folds.

### Ways to strengthen novelty (ordered by impact/effort)

1. **Run the neural arm** (already decided) → enables neural-only vs feature-only vs hybrid under identical grouped folds + "does the hybrid dilute explanation faithfulness?" (SHAP share of neural_* vs interpretable branches). No published ASAG work does this comparison leakage-aware. *(effort: 1 Colab day)*
2. **Add one LLM zero-shot baseline** (e.g., an open-weight model on the same test items, same metrics) → converts "no LLM comparison" from rejection risk to selling point. *(effort: 1–2 days + API/GPU cost)*
3. **Report the leakage audit on SemEval/SAF official splits too** (UA vs UQ/UD gap: SemEval 0.508→0.427/0.453; SAF 0.904→0.019) as "official-split corroboration" — data already exists in results.json. *(effort: hours — table only)*
4. Negation-window ablation (2/4/8) to defend assumption #2. *(effort: hours)*
