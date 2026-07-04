# Phase 8 — Journal Review Simulation · Phase 9 — Publication Readiness · Phase 10 — Roadmap

Simulated against the **current manuscript state** (hybrid pending). Re-run mentally after the Colab run.

---

## Phase 8 — Three anonymous reviewers

### Reviewer A (educational-NLP specialist, BEA/AIED community)

**Major strengths**
1. The leakage audit is genuinely useful and, to my knowledge, not published for Mohler/Powergrading/MIND-CA; the question-shortcut control (question identity alone ≥ full model under stratified CV on Powergrading) is a memorable, citable result.
2. Explanation validation against SAF's shipped gold feedback is a clever, annotation-free design; monotonicity across all four signals is convincing as far as it goes.
3. Honest reporting culture throughout (refusing head-to-head vs published numbers on subset corpora; declaring the SAF collapse; negative LODO result).

**Major weaknesses**
1. The headline accuracies are low (SemEval UD 0.427 macro-F1; ASAP subset QWK 0.363 vs human 0.942; MIND-CA 0.077 QWK). The paper argues these are *honest*, but the reviewer will ask whether a stronger model under the same honest protocol would change the story — the hybrid arm is promised, not delivered. **[resolved if hybrid table is filled]**
2. Condor et al. (2021) and the SemEval UA/UQ/UD design anticipate the central idea; the paper must be crisper that the novelty is the *audit + control baseline + significance machinery*, not unseen-question evaluation per se.
3. SAF explanation-validation effect sizes are moderate (ρ≤0.26, AUC≤0.62); claims should stay carefully scoped (they currently are, mostly).

**Minor:** rubric proxy = reference sentences needs earlier, franker flagging; negation window=4 unjustified; no teacher-facing evaluation of explanations.

**Verdict: Major revision. Acceptance probability: 55–65% (Q2/Q3) after revision; +10–15 pts with hybrid results.** Score: 6/10.

### Reviewer B (ML methodology / statistics)

**Major strengths**
1. Cluster bootstrap over questions with Holm correction is exactly right for this data structure and rare in this literature; reversing two apparent wins is the kind of self-critical result I trust.
2. HPO hygiene (dev/inner-CV only), materialized folds, seeds, controls — above-average rigor.
3. η² as a leakage-vulnerability diagnostic is a nice touch.

**Major weaknesses**
1. Significance uses seed-42 point predictions while headline tables use 5-seed means; defensible (stated in caption) but should be unified or justified more prominently.
2. No nested CV for the k-fold datasets' HPO — the "documented mild optimism" needs a bound estimate (the tuned-vs-default gap partially serves; make it explicit).
3. Perturbation results are single-run point estimates on n=300 with no CIs; and only lexical/negation features are recomputed — the "conservative bound" argument is fine but quantify what fraction of attribution mass the frozen features hold.
4. LODO's midpoint binarization is arbitrary; ASAP AUC 0.27 may be an artifact of that choice as much as of transfer — needs a sensitivity check (e.g., threshold at median).

**Minor:** report exact bootstrap CI method (percentile), n_boot convergence; runtime/memory table missing; "38 macro-F1 points" phrasing should be "0.38 absolute".

**Verdict: Major revision, but sympathetic. Acceptance probability: 60%.** Score: 6.5/10.

### Reviewer C (skeptical, LLM-era)

**Major strengths**
1. Deployment framing (CPU-only, calibrated, deferral curves, exact attributions) is a real institutional need that LLM-graders don't currently meet.
2. The protocol contribution transfers to LLM evaluation unchanged — this is the paper's best defense of relevance.

**Major weaknesses**
1. **No LLM baseline in 2026 is hard to accept.** Cheap to add (zero-shot on the six test sets under the same metrics); its absence invites rejection at first pass. The related-work citations (Chamieh 2024; LAK 2025) argue LLMs underperform — so show it under *your* protocol.
2. A 31-feature GBM is not a competitive grader by today's standards; without the hybrid numbers, the paper reads as an evaluation paper wearing a system paper's title.
3. If the hybrid lands and the DeBERTa arm dominates, the interpretability story needs the "attribution dilution" analysis to remain the point.

**Minor:** title promises "feature fusion" — consider retitling toward the audit if hybrid stays pending.

**Verdict: Reject in current state at a strong Q2; Accept-with-revisions at Q3 / after hybrid+LLM rows at Q2. Probability now: 35–45%; after additions: 65–75%.** Score: 5/10 now, 7/10 projected.

### Consensus editor summary
Publishable core (audit + validated explanations + statistical rigor) with two conditional gaps: hybrid results (promised by the paper itself) and an LLM baseline row. With both, this is a solid Q2 empirical/methodology paper; without, target Q3 or reframe the title/abstract fully around the audit.

---

## Phase 9 — Readiness scores (1–10)

| Dimension | Now (GBM-only, hybrid pending) | After hybrid + LLM row + minor fixes |
|---|---|---|
| Novelty | 6 (audit + SAF validation mechanism) | 7 |
| Scientific contribution | 6.5 | 7.5 |
| Methodology | 8 | 8.5 |
| Experiments | 6 (breadth yes; hybrid/LLM/runtime missing) | 8 |
| Writing | 7.5 (manuscript v2; needs native-speaker pass) | 8 |
| Figures | 6.5 (functional PNGs; need vector/consistent styling) | 7.5 |
| Statistical validity | 8.5 | 8.5 |
| Reproducibility | 8 (code+tests+checksums; repo URL missing) | 9 |
| References | 7.5 (verified core; ~12 flagged VERIFY) | 8.5 |
| **Overall quality** | **6.5–7** | **7.5–8** |

### Acceptance probability estimates (submission-ready = after roadmap "critical" items)

| Venue class | Now | After critical items |
|---|---|---|
| Scopus Q3 (e.g., Discover AI, Frontiers CS) | 55–65% | 75–85% |
| Scopus Q2 (e.g., KAIS, Applied Sciences, IEEE Access) | 35–45% | 60–70% |
| SCI/Q1 (EAAI, C&E:AI, IEEE TLT) | 10–15% | 25–35% |
| IEEE (Access) | 40% | 65% |
| Springer (KAIS, Discover AI, EAIT) | 40–50% | 65–75% |
| Elsevier (EAAI, ISWA) | 25–35% | 45–55% |
| MDPI (Applied Sciences, Information) | 60% | 80%+ |

*Estimates assume competent cover letter, venue-matched template, and no desk-reject triggers (scope match checked). These are calibrated judgments, not guarantees.*

---

## Phase 10 — Prioritized roadmap

### Critical (blocks submission)
| # | Task | Effort | Impact |
|---|---|---|---|
| C1 | **Run the DeBERTa hybrid on Colab** (`notebooks/02_neural_colab.ipynb` → `neural_oof.parquet` ×6) then regenerate: train2d, ablations (−D/only-D), xai (attribution-dilution), three-way table | 1 day GPU + 0.5 day CPU re-runs | Completes the paper's own promise; biggest single de-risk |
| C2 | Fill all `\TBD{}` slots in main.tex from the regenerated reports; update abstract/discussion/conclusion | 2–3 h | — |
| C3 | Verify the 12 "% VERIFY" bib entries (open each URL, fix authors/pages) | 2 h | Integrity |
| C4 | Release repo (or anonymized archive) + fill repository URL; fix CITATION.cff placeholder | 1–2 h | Reviewers check |
| C5 | Runtime/memory table (train+infer per dataset on the CPU; one script) | 2–3 h | Removes a guaranteed reviewer ask |

### Important (large expected review-score gain)
| # | Task | Effort | Impact |
|---|---|---|---|
| I1 | **LLM zero-shot baseline row** (one open-weight model and/or GPT-4o; same test items/metrics/protocol; temperature 0; log prompts) | 1–2 days | Converts Reviewer-C's rejection trigger into a headline discussion |
| I2 | Perturbation CIs (bootstrap over the 300-sample) + report frozen-feature attribution share | 0.5 day | Fixes Reviewer-B item 3 |
| I3 | LODO threshold sensitivity (midpoint vs median) | 2–3 h | Defuses the ASAP-inversion objection |
| I4 | Negation-window ablation {2,4,8} | 0.5 day | Closes a stated hole |
| I5 | Native-English editing pass + venue template swap | 1 day | Writing score |

### Quick wins (<2 h each)
- Add "0.38 absolute" phrasing consistency; unify seed-42 vs 5-seed footnote into Setup section.
- Export key figures as PDF/SVG with a consistent matplotlib style + larger fonts.
- Add per-item predictions dump to the release (enables future split-matched comparison — already claimed in the paper).
- Cover-letter draft naming the audit as primary contribution.

### Optional / long-term
- Teacher-facing user study of concept-coverage explanations (next paper).
- CORAL/CORN ordinal heads (genuine gap; torch slice already scaffolded).
- Widen ASAP-SAS to 10 prompts (Kaggle acceptance) + full Mohler 2011 corpus for split-matched SOTA comparison.
- Multilingual (SAF German half).
- Propose the audit as a standard reporting artifact (workshop paper / negative-results venue).

### Venue strategy
1st choice **Knowledge and Information Systems** (fit: GradeAid precedent, methods+evaluation papers, Q2) → 2nd **IEEE Access** (fast, broad) → 3rd **Applied Sciences** (fallback, high acceptance). If reframed education-first: Education & Information Technologies. Verify current quartiles at submission.

---

## Colab run — user instructions (C1)

1. Zip the project for Colab: `python experiments/make_colab_zip.py` (creates the code+data bundle).
2. Open `notebooks/02_neural_colab.ipynb` in Colab, Runtime → GPU (T4 ok).
3. Upload the zip, run all cells; it writes `data/processed/<name>/neural_oof.parquet` for each dataset (~15 min/dataset on T4).
4. Download the six parquets back into the same local paths.
5. Hand the block below to Claude Code.

## Claude Code handoff — post-Colab regeneration (C2)

```
CONTEXT
Repo: E:\master\فصل ثاني\NLP\exam\Explainable-Hybrid-ASAG (Windows; path contains Arabic
characters — CRITICAL: read CLAUDE.md "Critical Windows-Specific Gotchas" first and obey
all of them: venv at C:/Users/MSI/.cache/asag-venvs/asag-py311, never editable-install,
always PYTHONUTF8=1 PYTHONIOENCODING=utf-8, reinstall non-editable after any src/asag edit,
keep >=1-2 GB free on C: before any SBERT-touching run).

I have just run notebooks/02_neural_colab.ipynb on Colab and placed
data/processed/<name>/neural_oof.parquet for the six datasets (semeval, saf, asap_sas,
mohler, powergrading, mindreading).

TASK
Regenerate every result that changes when neural_* features exist, then fill the
placeholders in paper/main.tex. Do NOT touch feature engineering or splits.

CRITICAL CONTEXT — auto-concat trap: src/asag/models/data.py load_bundle AUTO-CONCATS
neural_* columns whenever neural_oof.parquet exists. Every downstream re-run (train,
train2d, xai, robustness, lodo, leakage_audit) silently becomes HYBRID and overwrites
the feature-only JSONs that paper/main.tex already quotes (verified by
paper/verify_numbers.py). The feature-only evidence base MUST be committed to git
BEFORE any re-run, and lodo / leakage_audit / train (2C) must NOT be re-run at all.

STEPS (verify each precondition before acting; if any check fails, STOP and report)
0. VERIFY a git commit exists that contains the current reports/** (feature-only
   evidence base). If uncommitted changes exist, commit them first:
   git add -A && git commit -m "feature-only evidence base (pre-hybrid)"
1. VERIFY the six neural_oof.parquet files exist, row counts match each
   features.parquet, and question_id order matches (data.load_bundle asserts this —
   run: python -c "from asag.models.data import load_bundle; [print(n,
   load_bundle(n).X.shape) for n in ['semeval','saf','asap_sas','mohler',
   'powergrading','mindreading']]" with the venv python and UTF-8 env vars).
   Expect 33 features everywhere (31 + neural_score + neural_pred).
2. Run the three-way comparison (feature-only / neural-only / hybrid), CLI confirmed
   at src/asag/models/neural_compare.py:209 (accepts optional dataset names):
   python -m asag.models.neural_compare
   -> reports/phase_hybrid/three_way.json + three_way.md
   -> reports/figures/phase_hybrid_three_way.png
3. Re-run: make ablations (must now produce -D and only-D variants; verify branch D
   is non-empty in reports/phase3/ablations.json). NOTE the semantics change: the new
   "full" = 33 features (hybrid); the feature-only anchor is now the "-D" variant and
   should match the OLD "full" (git show HEAD:reports/phase3/ablations.json) within
   seed noise (±0.01). If it does not, STOP and report.
4. Re-run: make xai (TreeSHAP now includes neural_*; extract the share of global
   |SHAP| absorbed by neural_* per dataset — needed for the "explanation dilution"
   sentence). Feature-only XAI numbers remain quotable from the STEP-0 commit.
5. DO NOT re-run make train2d by default: it would re-tune WITH branch D and
   overwrite reports/phase2d/results.json — the source of the paper's Table "main"
   (tuned feature-only). Only run it if the author explicitly wants a tuned-hybrid
   headline, in which case Tables main/sig must be re-framed and the old values
   preserved from the STEP-0 commit.
6. DO NOT re-run make lodo or make audit (neural_* would enter the shared-feature
   intersection and change their semantics). make robustness is optional; if run,
   note it becomes hybrid robustness and label it as such in the paper.
7. VERIFY integrity: reports/phase2d/results.json and phase4_lodo/lodo.json are
   byte-identical to the STEP-0 commit (git diff --stat HEAD -- reports/phase2d
   reports/phase4_lodo must be empty). If not, restore them from the commit before
   editing the paper.
8. Edit paper/main.tex: search for every "\TBD{" marker (10 sites) and fill from the
   fresh JSONs: Table hybrid (three-way + delta), abstract hybrid sentence,
   Sections 2.4 / 7.4 / 7.8 / 8 / 9 / Conclusion hybrid sentences, and the -D/only-D
   rows in Table ablations. Quote 5-seed mean±std where available; keep every number
   traceable to a JSON path in the table caption.
9. Update paper/analysis/PHASE8-10_reviews_readiness_roadmap.md: mark C1/C2 done,
   re-score the "After" column if hybrid results materially change the story.
10. Run: make test  (all tests must pass). Then compile the paper if a TeX
    distribution exists (pdflatex main && bibtex main && pdflatex main && pdflatex main
    inside paper/); if no TeX on this machine, say so — do not install one silently.

ACCEPTANCE CRITERIA
- Six neural_oof.parquet consumed; three_way.json exists with all six datasets.
- Zero remaining "\TBD{" occurrences in paper/main.tex.
- Feature-only numbers in the paper unchanged (within ±0.01) from pre-hybrid values.
- All tests pass; a short summary lists every number that changed and its JSON source.

EDGE CASES
- If a dataset's neural extraction failed on Colab (missing parquet), proceed with the
  remaining datasets, leave that dataset's TBD cells as \TBD{colab rerun needed}, and
  list it in the summary.
- If ablations.py skips branch D ("no features in this branch"), the parquet columns
  are misnamed — check they start with "neural_" before debugging anything else.
- asap_sas/mindreading have no reference answer: the neural premise falls back to the
  question text (dataset.py) — this is expected, not a bug.
```
