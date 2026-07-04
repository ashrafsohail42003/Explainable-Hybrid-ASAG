# Phase 3 — Literature Review (2019–2026)

Verification legend: **[V]** = authoritative page confirmed (ACL Anthology / publisher / arXiv abs). **[V-s]** = existence+venue verified via authoritative search result; pull exact author list before citing. **[U]** = unverified — do not cite without checking. Nothing below is fabricated; items were located by web search in July 2026.

Research area: **educational NLP → automated content scoring → ASAG**, intersecting evaluation methodology, interpretability/XAI, and calibration.

---

## 3.1 Foundations & datasets

| Work | Venue | Role here | Status |
|---|---|---|---|
| Mohler, Bunescu & Mihalcea, *Learning to grade short answer questions…* | ACL 2011 | Mohler corpus; similarity-feature era | [V] |
| Dzikovska et al., *SemEval-2013 Task 7* | SemEval 2013 | SemEval Beetle/SciEntsBank; **defined UA/UQ/UD splits** — the field's own unseen-question warning | [V] |
| Basu, Jacobs & Vanderwende, *Powergrading* | TACL 2013 | Powergrading corpus (clustering framing) | [V] |
| Burrows, Gurevych & Stein, *Eras and trends of ASAG* | IJAIED 2015 | Canonical survey pre-neural | [V] |
| Sultan, Salazar & Sumner, *Fast and easy SAG* | NAACL 2016 | Strong feature-based baseline era | [V] |
| Riordan et al., *Investigating neural architectures for short answer scoring* | BEA 2017 | ASAP-SAS neural reference: **test mean QWK 0.723 over all 10 prompts** (0.732 Fisher; ~0.744 was dev) — quote carefully | [V] |
| Kovatchev et al., *What is on your mind?* | COLING 2020 | MIND-CA corpus | [V] |
| Filighera et al., *Your answer is incorrect… bilingual short answer feedback dataset* (SAF) | ACL 2022 | SAF corpus + gold feedback; UA/UQ splits | [V] |
| Hewlett Foundation ASAP-SAS | Kaggle 2012 (@misc) + Shermis, Educ. Assessment 2015 | ASAP-SAS source | [V] |
| Meyer, Breuer & Fürst, *ASAG2024: a combined benchmark* | SIGCSE Virtual 2024 | Unifies 7 ASAG datasets; source of our Mohler slice; must cite (prior multi-dataset unification) | [V] |

## 3.2 Transformer-era ASAG

- Sung, Dhamecha, Mukhi, *Improving SAG using transformer-based pre-training*, **AIED 2019** [V]; Sung et al., *Pre-training BERT on domain resources for SAG*, **EMNLP-IJCNLP 2019** [V] (two distinct papers — don't conflate).
- Camus & Filighera, *Investigating transformers for ASAG*, AIED 2020 [V].
- Haller, Aldea, Seifert & Strisciuglio, *Survey on automated SAG with deep learning*, arXiv 2204.03503, 2022 (preprint only) [V] — **explicitly concludes hybrid handcrafted+transformer works best** → key motivation cite.
- Putnikovic & Jovanovic, *Embeddings for ASAG: scoping review*, IEEE TLT 2023 [V-s] — embeddings-vs-features contribution "inconclusive" → supports keeping interpretable features.
- Bexte, Horbach & Zesch, *Similarity-based content scoring* (BEA 2022; Findings ACL 2023) [V-s] — defends reference-comparison architectures as classroom-suitable → supports branch A/C design.

## 3.3 Evaluation protocols, generalization & leakage (our primary axis)

- **Condor, Litster & Pardos, EDM 2021** [V-s] — random vs question vs bundle holdout for SBERT ASAG; sharp drops on question holdout. **Closest antecedent of our audit; must be cited prominently.**
- Horbach & Zesch, *Influence of variance in learner answers…*, Frontiers in Education 2019 [V-s] — conceptual anchor (question-specific variance).
- Li & Ng, *Conundrums in cross-prompt AES*, ACL 2024 [V] — essay-side protocol critique; simple features match neural SOTA cross-prompt — perfect precedent for "protocol critique + simple features."
- Funayama et al., *Cross-prompt pre-finetuning for SAS*, AIED 2023 / IJAIED 2025 [V-s].
- Aggarwal et al., *EngSAF dataset* ("I understand why I got this grade"), arXiv 2407.12818 (2024/25) [V-s] — reports UA 75.4% vs UQ 58.7% (~17-pt unseen-question cliff).
- S-GRADES, *Studying generalization of student response assessments*, arXiv 2603.10233 (2026) [V-s, authors unverified] — cross-domain ASAG transfer benchmark; adjacent to our LODO.
- *ASAG with LLMs: from memorization to reasoning*, LAK 2026 [V-s, authors unverified] — names the memorization problem we operationalize.
- **Gap confirmed:** no published *audit* quantifying stratified-k-fold inflation with question-shortcut controls on Mohler/Powergrading/MIND-CA (searched July 2026).

## 3.4 Explainable ASAG (our second axis)

- Poulton & Eliens, *Explaining transformer-based models for ASAG*, ICDTE 2021 (ACM) [V-s] — saliency/IG/SHAP on SemEval; agreement with human graders.
- **Tornqvist et al., *ExASAG*, BEA 2023** [V-s] — SHAP/IG + NL explanations, validated against recruited experts. **Nearest neighbor to our SAF validation** — differentiate: we use *pre-existing* gold feedback (SAF `verification_feedback`), no new annotation, quantitative (ρ/AUC/monotonicity).
- Li et al., *Distilling ChatGPT for explainable automated student answer assessment* (AERA), **Findings of EMNLP 2023** [V] — rationale distillation; also the source of our ASAP-SAS mirror (cite for both reasons).
- *Explainable automatic grading with neural additive models*, AIED 2024 / arXiv 2405.00489 [V-s] — glass-box competitor design; accuracy-interpretability trade-off framing.
- *Neuro-symbolic multi-domain ASAG with justification cues*, arXiv 2403.01811 (2024) [V-s].
- Ariely et al., analytic-rubric scoring in biology, IJAIED 2023 / JRST 2024 [V-s] — pedagogy-grounded rubric-category scoring.

## 3.5 LLM-era grading (must-cover for 2026 reviewers)

- Chamieh, Zesch & Giebermann, *LLMs in short answer scoring: limitations…*, BEA 2024 [V-s] — zero/few-shot LLMs underperform supervised heads → justifies a trained auditable head.
- *ASAG for Finnish with ChatGPT*, AAAI 2024 [V-s, confirm authors] — GPT-4 QWK≥0.6 in only 44% of one-shot settings; length effects.
- Kortemeyer, *GPT-4 on ASAG*, Discover AI 2024; PhysRevPER 2023 [V-s].
- Schneider et al., *Towards LLM-based autograding*, arXiv 2309.11508 / CSEDU 2024 [V-s].
- Ferreira Mello et al., *Does GPT-4 with prompt engineering beat traditional models?*, LAK 2025 [V-s] — traditional ML F1 94.5 vs GPT-4 85.3.
- Rodrigues et al., *Is GPT-4 fair? …*, Computers & Education: AI 2025 [V-s] — leniency but demographic consistency.
- Yancey et al., BEA 2023 [V-s] — GPT-4 ≤ XGBoost baseline on CEFR essays.
- 2025–26 wave (cite 1–2 max, verify first): LLMarking (L@S 2025), Grade Guard (arXiv 2504.01253), CHiL(L)Grader calibrated HITL grading (arXiv 2603.11957), LLM-confidence estimation (arXiv 2605.00200) [all U beyond title].

**Trend synthesis:** LLM graders are lenient, prompt-sensitive, unvalidated per-item, and costly; supervised heads still win on agreement metrics (Chamieh 2024; LAK 2025). This is the paper's license to field a transparent CPU-only head — but a zero-shot LLM row in our main table is the single best pre-submission addition.

## 3.6 Hybrid features + transformers (our third axis)

- **Del Gobbo et al., *GradeAid*, Knowledge & Information Systems 2023** [V-s] — TF-IDF + BERT cross-encoder similarity → regressors, question-based splits, all public ASAG datasets. **Closest hybrid antecedent — must cite and differentiate** (ours: OOF-honest neural feature, NaN-native GBM, TreeSHAP preserved, leakage audit).
- *Hybrid AES: deep embeddings + handcrafted + XGBoost*, Mathematics (MDPI) 2024 [V-s] — essay-side analogue.
- Ormerod & Kwako, *Automated text scoring for the GPU-poor*, arXiv 2407.01873 (2024) [V-s] — small open models + GBM ensembling; kindred cost framing.

## 3.7 Robustness, calibration, deferral (our fourth axis)

- Funayama et al., *Preventing critical scoring errors with confidence estimation*, ACL SRW 2020 [V-s] — founded the defer-to-human framing; cite for risk–coverage.
- Funayama et al., *Human-in-the-loop SAS cost/quality*, AIED 2022 [V-s].
- Filighera et al., *Fooling ASAG systems*, AIED 2020; *Cheating ASAG with adjectives/adverbs*, IJAIED 2023 [V-s] — motivates perturbation suite.
- Ding et al., *Don't take "nswvtnvakgxpm" for an answer*, COLING 2020 [V-s].
- Guo et al., ICML 2017 (temperature scaling) [V]; Geirhos et al., *Shortcut learning*, Nature MI 2020 [V].

## 3.8 Methods toolbox (all [V])

SBERT (Reimers & Gurevych 2019); DeBERTa (He et al. ICLR 2021) / DeBERTaV3 (He, Gao, Chen ICLR 2023 — 3 authors); LightGBM (Ke et al. 2017); Optuna (Akiba et al. 2019); SHAP (Lundberg & Lee 2017); TreeSHAP (Lundberg et al., Nature MI 2020); CORAL (Cao, Mirjalili & Raschka, PRL 2020 — title includes "with application to age estimation"); CORN (Shi, Cao & Raschka, PAA 2023); Holm 1979; Efron & Tibshirani 1993.

## 3.9 The gap we fill (honest synthesis)

1. Unseen-question evaluation is *known* (SemEval splits; Condor 2021; EngSAF) — but **no systematic audit** of how much the still-default stratified k-fold inflates results on the no-official-split ASAG corpora, with question-shortcut controls and η² diagnosis. → our Table "leakage audit".
2. Explanation validation exists via bespoke expert studies (ExASAG) or generated rationales (AERA) — **nobody validates interpretable-feature attributions against SAF's shipped gold feedback**. → our SAF study.
3. Hybrids exist (GradeAid) — **none compare neural-only / feature-only / hybrid under identical grouped unseen-question folds while preserving exact TreeSHAP attributions**. → our (pending) hybrid table.
4. Calibration work exists — **not under the unseen-question protocol** where confidence actually degrades. → our ECE/risk–coverage under grouped CV.
5. CORAL/CORN ordinal heads: no ASAG application found → genuinely open (deferred torch slice).

## 3.10 Venue scan (Q2/Q3-realistic, ASAG-active 2023–26)

Knowledge & Information Systems (GradeAid's venue), Applied Sciences (MDPI), IEEE Access, Discover Artificial Intelligence, Engineering Applications of AI (competitive), IEEE TLT, Education & Information Technologies, Frontiers in CS/Education. Quartiles shift yearly — verify at submission. Best fit for this manuscript's framing: **KAIS / IEEE Access / Discover AI**; best prestige-per-effort backup: Applied Sciences.
