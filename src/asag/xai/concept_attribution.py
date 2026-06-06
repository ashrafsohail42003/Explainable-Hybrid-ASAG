"""Phase 2F — rubric-concept coverage attribution (the pedagogical differentiator).

The Phase 2B ``rub_*`` features only keep aggregates (mean/min/max/coverage). Here
we reconstruct the **per-concept** breakdown — exactly the rubric.py logic, but
returning each concept's similarity instead of collapsing it — so an explanation
reads the way a teacher justifies a mark: *"covered concepts 1 & 3, missed 2"*.

A "concept" is a sentence of the reference answer (rule-based sentencizer), scored
by SBERT cosine against the student answer; ``covered`` iff similarity ≥ tau.
Only datasets with a reference answer are attributable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asag.config import DataConfig


def concept_coverage(refs: list[str], students: list[str], encoder, nlp, tau: float
                     ) -> list[list[dict]]:
    """Per-row list of ``{concept, similarity, covered}`` for each reference sentence."""
    concept_lists: list[list[str]] = []
    all_concepts: list[str] = []
    for doc in nlp.pipe([r or "" for r in refs]):
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        concept_lists.append(sents)
        all_concepts.extend(sents)

    s_emb = encoder.embed([s or "" for s in students])
    if all_concepts:
        encoder.embed(all_concepts)        # warm the shared cache once

    out: list[list[dict]] = []
    for i, sents in enumerate(concept_lists):
        if not sents:
            out.append([])
            continue
        c_emb = encoder.embed(sents)
        sims = c_emb @ s_emb[i]
        out.append([
            {"concept": sents[j], "similarity": round(float(sims[j]), 4),
             "covered": bool(sims[j] >= tau)}
            for j in range(len(sents))
        ])
    return out


def _truncate(text: str, n: int = 220) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


def attribute_examples(name: str, cfg: DataConfig, encoder, nlp,
                       n_examples: int = 6) -> dict:
    """Per-concept coverage tables for a few representative answers of ``name``."""
    path = cfg.paths.processed / name / "encoder.parquet"
    if not path.exists():
        return {"status": "missing", "reason": f"no encoder.parquet for {name}"}
    df = pd.read_parquet(path)
    if "reference_answer_enc" not in df.columns:
        return {"status": "no_reference", "reason": "no reference_answer_enc column"}

    has_ref = df["reference_answer_enc"].fillna("").astype(str).str.strip() != ""
    df = df[has_ref]
    if df.empty:
        return {"status": "no_reference", "reason": "all reference answers empty"}

    # spread the examples across the score range so we show good + weak answers
    sc = pd.to_numeric(df.get("score"), errors="coerce")
    df = df.assign(_s=sc).sort_values("_s", na_position="last").reset_index(drop=True)
    picks = np.linspace(0, len(df) - 1, num=min(n_examples, len(df))).round().astype(int)
    rows = df.iloc[picks]

    tau = cfg.features.rubric.tau
    cov = concept_coverage(rows["reference_answer_enc"].tolist(),
                           rows["student_answer_enc"].tolist(), encoder, nlp, tau)
    examples = []
    for (_, r), concepts in zip(rows.iterrows(), cov):
        n_cov = sum(c["covered"] for c in concepts)
        examples.append({
            "question_id": str(r.get("question_id", "")),
            "score": _num(r.get("score")),
            "student_answer": _truncate(r["student_answer_enc"]),
            "n_concepts": len(concepts),
            "n_covered": int(n_cov),
            "coverage_fraction": round(n_cov / len(concepts), 3) if concepts else None,
            "concepts": concepts,
        })
    return {"status": "ok", "tau": tau, "n_examples": len(examples), "examples": examples}


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, 4)
