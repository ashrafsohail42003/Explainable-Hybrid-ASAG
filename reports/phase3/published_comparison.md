# Phase 3 — Head-to-head vs published (verify before submission)

> Published numbers are author recollections flagged `needs_verification`.
> **Claim?** = is a direct "we match/trail X" sentence valid (metric + data +
> split all aligned)? Currently **no dataset** clears that bar — every row is
> reported for *context only*. **Comparability** states why.

| Dataset | Metric | Ours (2C / 2D / neural) | Published (cite) | Claim? | Comparability |
|---|---|---|---|---|---|
| semeval | macro_f1 | 0.4111 / 0.4269 / — | SemEval-2013 shared task, best 5-way macro-F1 ~0.55-0.62 [Dzikovska et al. 2013, S13-2045]; BERT fine-tuned, SciEntsBank 5-way ~0.58 [Sung et al. 2019] | ❌ context-only | context-only — split must be matched (we headline the hard test_ud) |
| asap_sas | qwk | 0.3497 / 0.3630 / — | Neural LSTM/attention, ASAP-SAS mean QWK ~0.74 [Riordan, Horbach et al. 2017, BEA W17-5017]; Kaggle ASAP-SAS private LB top ~0.78 [Kaggle 2012 (not our prompt set)] | ❌ context-only | not comparable — 4/10 prompts (AERA mirror) vs all-10 published |
| mohler | pearson | 0.4297 / 0.4904 / — | Mohler et al. 2011 best r~0.52 [Mohler et al. 2011] | ❌ context-only | not comparable — ASAG2024 subset (21q) vs full corpus (80q) |
| saf | pearson | 0.0261 / 0.0193 / — | SAF baseline RMSE-based [Filighera et al. 2022] | ❌ context-only | metric-mismatch — published reports RMSE/feedback-F1 |
| powergrading | macro_f1 | 0.4500 / 0.4862 / — | Powergrading clustering clustering metrics [Basu et al. 2013] | ❌ context-only | setup-mismatch — original is clustering, not supervised F1 |
| mindreading | qwk | 0.0594 / 0.0770 / — | MIND-CA baselines accuracy/F1 reported [Kovatchev et al. 2020] | ❌ context-only | approximate — same corpus, check exact metric/setup |

## Per-dataset notes

- **semeval** — SciEntsBank+Beetle 5-way. Our headline is the unseen-domain test_ud (the hardest split); most published 5-way macro-F1 are on unseen-answers (test_ua). A claim is only valid split-matched — report our test_ua next to test_ua numbers, not test_ud.
- **asap_sas** — EssaySets 1/2/5/6 only (the free AERA mirror). Published QWK is Fisher-averaged over all 10 prompts, so it is NOT our prompt set; the Kaggle LB (~0.78, all 10) is likewise off-limits.
- **mohler** — report as an internal number only; do not claim vs Mohler-2011
- **saf** — we report Pearson; SAF paper centers RMSE + verification-feedback F1. We treat SAF as the explainability case study (it uniquely ships gold feedback), not an accuracy headline — our test_uq Pearson is ~0 and the gain is NOT significant (see significance.json).
- **powergrading** — Basu 2013 frames it as answer clustering; our supervised F1 is a different task. NB: under the cluster (question-level) bootstrap the head's gain over baseline is NOT significant (only 20 questions).
- **mindreading** — Kovatchev 2020 reports accuracy/F1 on the 0/1/2 task; we report QWK
