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
