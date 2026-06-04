"""Rubric concept-coverage features (``rub_*``).

No curated rubric/key-concept lists exist in any dataset, so we use a
**sentence-level proxy**: each sentence of the reference answer is a "concept".
Concepts are obtained with the rule-based sentencizer (no parser), embedded with
the shared SBERT encoder, and scored by cosine against the student answer.

  * ``rub_n_concepts``      — number of reference sentences
  * ``rub_mean_maxsim``     — mean per-concept similarity
  * ``rub_min_maxsim``      — weakest covered concept
  * ``rub_max_maxsim``      — strongest covered concept
  * ``rub_coverage_at_tau`` — fraction of concepts with sim ≥ tau

All ``NaN`` when there is no reference (single-sentence references make these
collapse toward ``sem_cosine`` — documented).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asag.features.text_utils import NAN

COLUMNS = ["rub_n_concepts", "rub_mean_maxsim", "rub_min_maxsim",
           "rub_max_maxsim", "rub_coverage_at_tau"]


def compute_rubric(df: pd.DataFrame, cfg, encoder, nlp) -> pd.DataFrame:
    refs = df["reference_answer_enc"].fillna("").astype(str).tolist()
    students = df["student_answer_enc"].fillna("").astype(str).tolist()
    tau = cfg.features.rubric.tau
    batch = cfg.features.semantic.batch_size

    concept_lists: list[list[str]] = []
    all_concepts: list[str] = []
    for doc in nlp.pipe(refs, batch_size=batch):
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        concept_lists.append(sents)
        all_concepts.extend(sents)

    s_emb = encoder.embed(students)           # (n, d), normalized
    if all_concepts:
        encoder.embed(all_concepts)           # warm the shared cache

    rows: list[dict] = []
    for i, sents in enumerate(concept_lists):
        if not sents:
            rows.append({c: NAN for c in COLUMNS})
            continue
        c_emb = encoder.embed(sents)          # (k, d)
        sims = c_emb @ s_emb[i]               # normalized -> cosine
        rows.append({
            "rub_n_concepts": float(len(sents)),
            "rub_mean_maxsim": float(sims.mean()),
            "rub_min_maxsim": float(sims.min()),
            "rub_max_maxsim": float(sims.max()),
            "rub_coverage_at_tau": float((sims >= tau).mean()),
        })

    return pd.DataFrame(rows, columns=COLUMNS, index=df.index)
