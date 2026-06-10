"""Trivial reference predictors for ΔQWK / Δmetric context.

A learned head only earns its keep if it beats these. The report specifically
warns about Mohler's skew toward the top grade (a constant-high predictor can
look deceptively strong), so a constant-max baseline is exposed too.
"""

from __future__ import annotations

import numpy as np


def fit_predict_baseline(y_train: np.ndarray, n_test: int, task_type: str) -> np.ndarray:
    """Constant predictor calibrated on the training target.

    * classification → majority class code
    * ordinal        → rounded median grade
    * regression     → training mean
    """
    yt = np.asarray(y_train, dtype=float)
    yt = yt[np.isfinite(yt)]
    if yt.size == 0:
        return np.full(n_test, np.nan)
    if task_type == "classification":
        vals, counts = np.unique(yt, return_counts=True)
        const = vals[int(np.argmax(counts))]
    elif task_type == "ordinal":
        const = float(np.rint(np.median(yt)))
    else:  # regression
        const = float(np.mean(yt))
    return np.full(n_test, const, dtype=float)


def constant_max(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Predict the maximum training grade everywhere (skew-foil baseline)."""
    yt = np.asarray(y_train, dtype=float)
    yt = yt[np.isfinite(yt)]
    if yt.size == 0:
        return np.full(n_test, np.nan)
    return np.full(n_test, float(np.max(yt)), dtype=float)


def question_shortcut_predict(
    y_train: np.ndarray, qid_train, qid_test, task_type: str,
) -> np.ndarray:
    """Per-question memorization probe: predict each test row from its own
    question's training target alone (mean for reg/ordinal, majority for clf).

    This is the reviewer-facing leakage control. Under a question-leaked split
    (legacy stratified k-fold, or SAF ``test_ua``) it scores high — the question
    is in train. Under a grouped/unseen-question split the question is absent, so
    it falls back to the global constant and collapses. The gap between the two
    is the size of the memorization shortcut.
    """
    yt = np.asarray(y_train, dtype=float)
    qtr = np.asarray(qid_train).astype(str)
    qte = np.asarray(qid_test).astype(str)
    finite = np.isfinite(yt)
    yt, qtr = yt[finite], qtr[finite]
    if yt.size == 0:
        return np.full(len(qte), np.nan)

    if task_type == "classification":
        glob = fit_predict_baseline(yt, 1, task_type)[0]
        per_q: dict[str, float] = {}
        for q in np.unique(qtr):
            vals, counts = np.unique(yt[qtr == q], return_counts=True)
            per_q[q] = float(vals[int(np.argmax(counts))])
    else:
        glob = float(np.mean(yt))
        per_q = {q: float(np.mean(yt[qtr == q])) for q in np.unique(qtr)}
        if task_type == "ordinal":
            per_q = {q: float(np.rint(v)) for q, v in per_q.items()}
            glob = float(np.rint(glob))
    return np.array([per_q.get(q, glob) for q in qte], dtype=float)
