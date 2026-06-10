# Paper (Phase A draft)

Short-paper draft for the **feature-only** (GBM) version of the project, reframed
around the leakage-aware evaluation and interpretability contributions. The
neural cross-encoder arm is deferred to the extended version (Phase B).

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`main.tex` uses the stock `article` class so it compiles anywhere. Figures are
pulled from `../reports/figures/` via `\graphicspath`.

## Before submission — checklist

- [ ] **Swap the template** to the target venue (ACL `acl.sty` bundle for
      LREC-COLING / \*SEM, or the IEEE class for a Q2 journal). Section structure
      and `\cite` keys carry over unchanged.
- [ ] **Verify every citation** in `references.bib` against the source paper —
      authors, venue, year, and especially the *numbers* quoted in
      `reports/phase3/published_comparison.md` (all flagged `needs_verification`).
      ASAP-SAS (Riordan et al. 2017, ~0.74 Fisher-mean over **all 10** prompts)
      and SemEval (Dzikovska et al. 2013) are the only near-comparable anchors,
      and even those are **not** split-matched to our headline — do not claim a
      head-to-head.
- [ ] Confirm all numbers in Tables 2–4 still match the regenerated reports
      (`reports/phase2d/`, `reports/phase3/`, `reports/phase4_*`).
- [ ] Add author affiliation / acknowledgements.
- [ ] Phase B: add the DeBERTa cross-encoder results (transformer-only /
      feature-only / hybrid) and update the abstract, Table 3, and Section 8.

## Figures used

| Figure | Source |
|---|---|
| `phase2d_significance.png` | cluster-bootstrap CIs (Holm-coloured) |
| `phase3_branch_delta.png`  | branch ablation deltas |
| `phase2f_saf_validation.png` | SAF explainability validation |
| `phase4_calibration.png`   | reliability pre/post temperature scaling |
