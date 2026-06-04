"""TF-IDF cosine between student and reference (``tfidf_cosine``).

A ``TfidfVectorizer`` is fit per dataset on the union of student + reference
feature-view text; cosine is the dot product of the L2-normalized rows. ``NaN``
for rows without a reference (and for whole datasets that have none).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from asag.features.text_utils import NAN

COLUMNS = ["tfidf_cosine"]


def compute_tfidf(df: pd.DataFrame, cfg) -> pd.DataFrame:
    tcfg = cfg.features.tfidf
    students = df["student_answer_feat"].fillna("").astype(str).tolist()
    refs = df["reference_answer_feat"].fillna("").astype(str).tolist()
    has_ref = np.array([bool(r.strip()) for r in refs])

    if not has_ref.any():  # dataset has no reference answers at all
        return pd.DataFrame({"tfidf_cosine": np.full(len(df), NAN)}, index=df.index)

    n = len(df)
    corpus = students + refs
    kwargs = dict(
        ngram_range=(tcfg.ngram_min, tcfg.ngram_max),
        token_pattern=r"(?u)\b\w+\b",
    )
    try:
        X = TfidfVectorizer(min_df=tcfg.min_df, **kwargs).fit_transform(corpus)
    except ValueError:  # min_df pruned the whole vocabulary -> retry unpruned
        X = TfidfVectorizer(min_df=1, **kwargs).fit_transform(corpus)

    sx, rx = X[:n], X[n:]
    # rows are L2-normalized by TfidfVectorizer -> cosine == elementwise dot
    cos = np.asarray(sx.multiply(rx).sum(axis=1)).ravel().astype(np.float64)
    cos[~has_ref] = NAN
    return pd.DataFrame({"tfidf_cosine": cos}, index=df.index)
