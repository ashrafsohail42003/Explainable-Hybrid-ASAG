# Leakage-Aware, Interpretable Feature Fusion for Automatic Short Answer Grading

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/main_ieee.pdf)

A research-grade, **explainable hybrid** Automatic Short Answer Grading (ASAG) system evaluated
honestly across **six heterogeneous datasets**. The project pairs a rigorous **question-leakage
audit** with an **interpretable gradient-boosted fusion grader** and an **explainability study
validated against human gold feedback** — and fuses a DeBERTa cross-encoder as an *out-of-fold,
fully attributable* feature.

> **Paper:** [`paper/main_ieee.pdf`](paper/main_ieee.pdf) &nbsp;•&nbsp;
> **Authors:** Ashraf Sohail Alkahlout, Abdulaziz Mahmoud Lubbad &nbsp;•&nbsp;
> **Supervisor:** Prof. Aiman Ahmed Abu Samra &nbsp;•&nbsp;
> Department of Computer Engineering, Islamic University of Gaza

---

## Highlights

- **Evaluation honesty (RQ1).** Replacing stratified *k*-fold with **grouped leave-questions-out**
  cross-validation collapses apparent performance by up to **38 macro-F1 points**
  (Powergrading `0.830 → 0.450`). A control model that sees **only the question identity** scores
  `0.873` under the leaky protocol — *above* the full model — exposing question memorization.
- **Interpretable fusion grader (RQ2).** A NaN-native LightGBM head over lexical, semantic, and
  rubric-coverage branches spans classification, ordinal, and regression targets. Linguistic
  features are the accuracy workhorse; significance is established by a **question-clustered
  bootstrap under Holm correction** (significant on 4/6 datasets).
- **Faithful, human-validated explanations (RQ3).** Exact TreeSHAP rankings match gain importance
  (Spearman ρ = 0.65–0.999); rubric coverage rises monotonically with SAF human verdicts
  (Incorrect `0.224` → Partial `0.476` → Correct `0.547`).
- **Deployment (RQ4).** Temperature scaling halves SemEval ECE (`0.122 → 0.059`); perturbation and
  leave-one-dataset-out transfer reported.
- **Transformer-as-feature hybrid (RQ5).** A DeBERTa-v3 cross-encoder fused as an out-of-fold
  feature gives a large, significant gain where its signal is discriminative
  (**Powergrading `+0.231` macro-F1**), while preserving exact TreeSHAP attributions.

## Headline results (grouped leave-questions-out)

| Dataset | Metric | Feature-only | Hybrid | Tuned | Significant |
|---|---|---|---|---|:--:|
| SemEval-2013 | macro-F1 | 0.415 | 0.427 | 0.430 | ✅ |
| ASAP-SAS | QWK | 0.340 | 0.383 | 0.385 | ✅ |
| Mohler | Pearson | 0.439 | 0.425 | 0.488 | ✅ |
| Powergrading | macro-F1 | 0.473 | **0.704** | **0.729** | ✅ |
| SAF | Pearson | 0.024 | 0.026 | 0.007 | — |
| MIND-CA | QWK | 0.063 | 0.017 | −0.009 | — |

ASAP-SAS human inter-annotator ceiling: **QWK 0.942** (the model is honestly below human parity).

## Repository structure

```
src/asag/           # library: data, features, models, neural, xai
configs/data.yaml   # single source of truth (paths, branches, model, neural)
notebooks/          # 01 EDA; 02 neural (Colab); Kaggle variant
experiments/        # LLM zero-shot baseline, audit + migration scripts
paper/              # main_ieee.tex + main_ieee.pdf + references.bib
reports/            # all JSON reports + figures (phase2a…phase4, phase_hybrid)
tests/              # pytest suites (loaders, features, models, xai, neural)
Makefile            # reproducible pipeline targets
```

## Reproducing the results

The interpretable pipeline is CPU-only and needs no GPU:

```bash
make setup        # Python 3.11 env + deps + spaCy model
make download     # acquire the six datasets under their own licenses
make preprocess   # two-view preprocessing
make features     # build interpretable feature matrices
make train        # Phase 2C: LightGBM fusion head
make train2d      # Phase 2D: Optuna HPO + cluster-bootstrap significance
make ablations    # Phase 3: branch ablations (A/B/C/D + negation)
make xai          # Phase 2F: TreeSHAP + concept coverage + SAF validation
make audit lodo robustness   # Phase 4: leakage audit, LODO, calibration
```

The DeBERTa hybrid arm needs a GPU; run it on Colab/Kaggle/RunPod (see
[`notebooks/02_neural_colab.ipynb`](notebooks/02_neural_colab.ipynb)), which writes
`data/processed/<ds>/neural_oof.parquet` for the LightGBM head to auto-fuse.

## Datasets

Six English ASAG corpora — SemEval-2013 Task 7, SAF, ASAP-SAS, Mohler (2011), Powergrading (2013),
and MIND-CA — spanning classification, ordinal, and regression targets. **Raw data is never
redistributed** here; each dataset is acquired under its own license (see
[`reports/DATASETS.md`](reports/DATASETS.md)).

## Citation

If you use this work, please cite (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{alkahlout2026leakage,
  title   = {Leakage-Aware, Interpretable Feature Fusion for Automatic Short
             Answer Grading across Heterogeneous Datasets},
  author  = {Alkahlout, Ashraf Sohail and Lubbad, Abdulaziz Mahmoud and
             Abu Samra, Aiman Ahmed},
  year    = {2026}
}
```

## License

Code released under the [MIT License](LICENSE). Dataset licenses vary and are the responsibility
of the user (see `reports/DATASETS.md`).
