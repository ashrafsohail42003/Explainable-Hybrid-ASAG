# Paper — full-length journal manuscript (Q2/Q3-ready, generic template)

`main.tex` is the **full-length manuscript** (rewritten 2026-07-04 from the earlier
short draft): leakage-aware evaluation + interpretable fusion + validated
explanations across six datasets, with the DeBERTa hybrid arm structured in but
**pending the Colab run** — every pending value is marked `\TBD{...}`.

`analysis/` contains the research-engineering companion documents:

| File | Content |
|---|---|
| `PHASE1-2_understanding_and_contribution.md` | Reverse-engineering of the pipeline, hidden assumptions, contribution & novelty analysis |
| `PHASE3_literature_review.md` | 2019–2026 literature by theme, verification status per citation, gap synthesis, venue scan |
| `PHASE4-6_technical_review_experiments_improvements.md` | Component-by-component technical review, missing-experiment map, ranked improvements |
| `PHASE8-10_reviews_readiness_roadmap.md` | 3-reviewer simulation, readiness scores, acceptance estimates, prioritized roadmap, **Colab instructions + Claude Code handoff block** |

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Stock `article` class — compiles anywhere; swap the preamble for the venue
template at submission (section structure and `\cite` keys carry over). Figures
pull from `../reports/figures/` via `\graphicspath`.

## Before submission — checklist

- [ ] **C1: Run the hybrid** — `notebooks/02_neural_colab.ipynb` on Colab →
      six `neural_oof.parquet` → regenerate (see the handoff block in
      `analysis/PHASE8-10...md`).
- [ ] **C2: Fill every `\TBD{}`** in `main.tex` (search for "TBD"; 0 must remain).
- [ ] **C3: Verify the `% VERIFY:` bib entries** in `references.bib` (12 flagged;
      open each URL, confirm authors/pages). Never cite the "CANDIDATE CITATIONS"
      block at the bottom without verification.
- [ ] **C4:** Fill repository URL (Reproducibility section + `CITATION.cff`).
- [ ] **C5:** Add the runtime/memory table (marked as TODO in Section 6).
- [ ] **I1 (strongly recommended):** add a zero-shot LLM baseline row.
- [ ] Swap template to target venue; add affiliation; native-English pass.
- [ ] Re-confirm all table numbers against regenerated `reports/**`
      (verification script: `paper/verify_numbers.py`).

## Numbers provenance

Every number in `main.tex` traces to a JSON under `reports/`:
Table leakage → `phase4_audit/leakage_audit.json`; Table main →
`phase2d/results.json`; Table sig → `phase2d/significance.json`; Table ablations
→ `phase3/ablations.json`; Table SAF → `phase2f/saf_validation.json`;
calibration/perturbations → `phase4_robust/robustness.json`; LODO →
`phase4_lodo/lodo.json`; ceiling → `phase2d/ceiling.json`; error analysis →
`phase3/error_analysis.json`; SHAP-gain ρ → `phase2f/shap.json`; tuned params →
`phase2d/results.json`.
