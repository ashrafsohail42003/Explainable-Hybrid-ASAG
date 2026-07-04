# Phase 4 — Technical Review · Phase 5 — Experiment Design · Phase 6 — Scientific Improvements

Verified against source + `reports/**`. Ratings: ✅ sound · ⚠️ acceptable with caveat · ❌ needs work.

---

## Phase 4 — Component-by-component review

| Component | Verdict | Evidence & notes |
|---|---|---|
| Dataset quality | ⚠️ | 6 corpora, 3 task types — breadth is a strength. Mohler is the ASAG2024 *subset* (21q/616 post-dedupe) and ASAP-SAS the AERA mirror (4/10 prompts) → internal numbers only (correctly documented). Dedupe (median-score per (q, answer)) removes label noise: Powergrading 13,960→4,941, MIND-CA 11,311→10,543, Mohler 1,260→616. |
| Class/score imbalance | ⚠️ | SemEval non_domain support=20/4,562; SAF Incorrect=63/854 in XAI slice; Mohler pos-rate 0.91 in LODO. Macro-F1/QWK partially compensate; imbalance never explicitly treated (no weights/resampling). Acceptable if stated. |
| Cleaning/preprocessing | ✅ | Two-view contract is a genuine strength (lossless encoder view; negation-preserving feature view; scope-marked third view). Enforced by tests. |
| Tokenization | ✅ | max_len=128 justified empirically (p99, truncation %) per tokenizer/regime; SAF override 256 documented. Better practice than most ASAG papers. |
| Embeddings | ⚠️ | SBERT all-MiniLM-L6-v2 (2019-era, 384-d, CPU). Fine as interpretable-signal source; weak as semantic ceiling — the pending DeBERTa arm addresses exactly this. |
| Classifier/head | ✅ | NaN-native LightGBM; ordinal = regression + round/clip (defensible, simpler than CORAL; CORN/CORAL deferred and flagged). Trivial + question-shortcut controls always reported. |
| Hyperparameters | ✅ | Optuna TPE, 40 trials, per-dataset; search space serialized; **objective never touches test** (dev split or inner grouped CV). Mild HPO optimism for k-fold case documented. Nested CV absent (⚠️, state as limitation). |
| Training/validation/testing | ✅ | Grouped LQO CV for no-official-split corpora; official splits respected; 5 seeds mean±std; materialized fold column (never re-split). |
| Metrics | ✅ | Task-appropriate per registry (macro-F1 / QWK / Pearson+RMSE); pooled-OOF; per-prompt QWK averaged (pooling trap avoided). |
| Statistical testing | ✅ | Cluster bootstrap resampling *questions* (10k, one-sided p(Δ≤0)) + Holm across 6 tests. ⚠️ uses seed-42 point predictions — must be stated (5-seed means differ in 3rd decimal). |
| Error analysis | ✅ | Full confusion matrices, per-class P/R/F1, top-confusions (SemEval: partial→correct 524 cases = the dominant error). |
| Ablations | ✅ | Branch −X/only-X + negation, fixed regularized head, 5 seeds — clean attribution. |
| Explainability | ✅ | Exact TreeSHAP (no shap dep); SHAP-vs-gain ρ 0.653–0.999; faithfulness decomposition (validity/use/sign-consistency/share); SAF gold-feedback validation. |
| Robustness | ⚠️ | 3 perturbations ×300 samples, seed-fixed, run once (no CI); only lexical+negation features recomputed (conservative — documented). Calibration clean (val slice ⊥ test; T grid 191 values). |
| Complexity/runtime | ❌ | **Missing**: no wall-clock/latency/memory numbers anywhere. Trivial to add; reviewers ask. |
| Reproducibility | ✅ | Seed 42 global; sha256 checksums; pinned deps; 73 tests; CITATION.cff. Repo URL placeholder empty (❌ fill before submission). |

**Top technical debts to fix pre-submission (all cheap):**
1. Runtime/memory table (train+inference per dataset, CPU spec).
2. State the single-seed significance choice in the caption.
3. Fill CITATION.cff repository URL + release code/splits (anonymized if needed).
4. One line justifying negation window=4 or add the 2/4/8 ablation.
5. Report perturbation deltas with a bootstrap CI (they're currently point estimates).

---

## Phase 5 — Experiment design (what a Q2/Q3 reviewer expects vs what exists)

| Expected experiment | Status | Action |
|---|---|---|
| Trivial baselines | ✅ done (majority/mean/median) | — |
| Shortcut control | ✅ done (question-shortcut arm) | highlight — it's novel-ish |
| SOTA comparison | ⚠️ context-only table exists (`published_comparison.md`), correctly refuses head-to-head (subset corpora) | keep; add split-matched SemEval test_ua row vs Sung et al. if verified |
| **LLM zero-shot baseline** | ❌ **missing — highest-value addition** | GPT-4o or open-weight (Llama-3.1-8B/70B) zero-shot on the same test items, same metrics, temperature 0; cost ~1–2 days. Converts the 2026 "why no LLM?" objection into a result. |
| **Hybrid three-way (neural/feature/hybrid)** | ❌ implemented, not run | Colab run → `phase_hybrid/three_way.json` + ablation branch D + faithfulness-dilution check |
| Cross-validation rigor | ✅ grouped LQO, 5 seeds | — |
| Statistical significance | ✅ cluster bootstrap + Holm | — |
| Confusion matrices | ✅ in error_analysis.json | move key one into paper |
| ROC / PR curves | ⚠️ only AUC in LODO | optional; risk–coverage curve already covers the deployment story better |
| Calibration | ✅ ECE + temperature + reliability diagrams | — |
| Robustness | ✅ 3 perturbations | add CI; optionally add Filighera-style adjective/adverb attack for one dataset |
| Generalization | ✅ UA/UQ/UD + generalization_gap + LODO | — |
| Human ceiling | ✅ ASAP IAA QWK 0.9419 | contextualize model 0.363 honestly (subset + unseen protocol) |
| Complexity/latency | ❌ missing | add table |
| Learning curves | ❌ missing | optional, nice for "data efficiency of features vs neural" once hybrid lands |

**Priority order for new experiments:** (1) hybrid run [decided], (2) LLM baseline, (3) runtime table, (4) perturbation CIs, (5) negation-window ablation, (6) learning curves.

**Why each matters:** (1) completes the paper's own promise (title says hybrid); (2) 2026 reviewer table-stakes; (3) supports the "CPU-only, deployable" claim; (4) turns anecdote into evidence; (5) closes an ablation hole a careful reviewer will probe; (6) strengthens the classroom-suitability argument (Bexte et al. line).

---

## Phase 6 — Scientific improvements (ranked, with expected impact)

1. **DeBERTa-v3 cross-encoder OOF hybrid** *(already implemented; run it)*. Expected: biggest headline movement on SemEval/SAF/Mohler (reference-answer datasets). Risk: SAF UQ may stay ≈0 (only 5 test questions) — that itself is a publishable observation about question diversity, not a failure.
2. **LLM zero-shot/few-shot row** — expected: LLM beats GBM on SemEval UD but with leniency bias and no calibration; either outcome strengthens the discussion (cite Chamieh 2024, LAK 2025 alignment).
3. **CORAL/CORN ordinal heads** (torch slice, deferred) — expected small QWK gains on ASAP/MIND-CA; also a genuine gap (no ASAG application found) → good "future work made concrete".
4. **Data augmentation for unseen-question robustness** (paraphrase reference answers, back-translation of student answers on train folds only). Expected: modest; keep as future work.
5. **Question-conditioned features** (question-type, expected-answer-length priors) — could reduce the MIND-CA floor; cheap.
6. **Active-learning / deferral policy** built on the risk–coverage curve (route lowest-confidence 20% to humans: SemEval accuracy 0.467→0.506 at 80% coverage — already measured) — frame as deployment guidance, no new experiment needed.
7. **Multi-task across datasets** (shared feature space already exists via LODO) — research-grade extension, not for this paper.
8. Prompt-engineering / RAG / knowledge augmentation — out of scope for the interpretable-head thesis; mention only in future work if LLM baseline added.

**Explicitly rejected improvements (with reason):**
- Replacing GBM with a deep fusion net: destroys the NaN-native + exact-TreeSHAP story for marginal accuracy.
- Tuning tau (rubric threshold) on dev: would contaminate the "untuned, honest" faithfulness narrative; leave fixed, state it.
- Pooling ASAP prompts into one model: known reviewer trap (per-prompt rubrics), already correctly avoided.
