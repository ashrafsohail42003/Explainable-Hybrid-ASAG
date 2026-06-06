"""Scoring metrics for Phase 2C.

QWK is the headline agreement metric for the ordinal datasets; regression
datasets use Pearson/Spearman/RMSE/MAE; classification uses macro/weighted F1.
All functions take 1-D array-likes and return plain ``float`` (``nan`` when the
metric is undefined, e.g. a constant prediction for a correlation).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def _as_int(y) -> np.ndarray:
    return np.rint(np.asarray(y, dtype=float)).astype(int)


def qwk(y_true, y_pred) -> float:
    """Quadratic weighted kappa on integer-rounded labels.

    Uses a shared label set so ``y_true``/``y_pred`` align; returns ``nan`` when
    fewer than two distinct labels are present (kappa undefined).
    """
    yt, yp = _as_int(y_true), _as_int(y_pred)
    labels = np.unique(np.concatenate([yt, yp]))
    if labels.size < 2:
        return float("nan")
    return float(cohen_kappa_score(yt, yp, weights="quadratic", labels=labels))


def rmse(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def _corr(y_true, y_pred, kind: str) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    # correlation is undefined when either side is constant
    if yt.size < 2 or np.allclose(yt.std(), 0.0) or np.allclose(yp.std(), 0.0):
        return float("nan")
    from scipy.stats import pearsonr, spearmanr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = pearsonr(yt, yp)[0] if kind == "pearson" else spearmanr(yt, yp)[0]
    return float(r)


def pearson(y_true, y_pred) -> float:
    return _corr(y_true, y_pred, "pearson")


def spearman(y_true, y_pred) -> float:
    return _corr(y_true, y_pred, "spearman")


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(_as_int(y_true), _as_int(y_pred), average="macro", zero_division=0))


def weighted_f1(y_true, y_pred) -> float:
    return float(f1_score(_as_int(y_true), _as_int(y_pred), average="weighted", zero_division=0))


def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(_as_int(y_true), _as_int(y_pred)))


_DISPATCH = {
    "qwk": qwk,
    "rmse": rmse,
    "mae": mae,
    "pearson": pearson,
    "spearman": spearman,
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1,
    "accuracy": accuracy,
}


def compute_metrics(y_true, y_pred, names) -> dict[str, float]:
    """Compute the named metrics; empty input yields all-``nan``."""
    yt = np.asarray(y_true)
    out: dict[str, float] = {}
    for name in names:
        fn = _DISPATCH.get(name)
        if fn is None:
            raise KeyError(f"unknown metric {name!r}; known: {sorted(_DISPATCH)}")
        out[name] = float("nan") if yt.size == 0 else fn(y_true, y_pred)
    return out
