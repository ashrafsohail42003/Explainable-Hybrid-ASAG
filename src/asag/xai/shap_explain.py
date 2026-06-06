"""Phase 2F — SHAP over the fusion head (the quantitative backbone).

Uses LightGBM's exact TreeSHAP (``pred_contrib``) — no ``shap`` dependency, no
``numba``/``llvmlite`` on the C:-constrained box. Produces, per dataset:

* a **global** ranking (mean |SHAP| per feature) — the model-faithful counterpart
  to the Phase 2C gain importance, with the rank correlation between the two; and
* a handful of **local** explanations (representative answers with the signed
  per-feature contributions that drove their grade).

Ordinal heads are explained on the raw regression output (before round-and-clip).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from asag.config import DataConfig
from asag.models.data import Bundle
from asag.xai.common import (
    fit_head_on_all, global_importance, load_tuned_params, shaped_contribs,
)


def _example_indices(y_pred: np.ndarray, task_type: str, k: int) -> list[int]:
    """Pick a few representative rows: spread by predicted grade, or one per class."""
    n = len(y_pred)
    if n == 0:
        return []
    if task_type == "classification":
        seen: dict[float, int] = {}
        for i, v in enumerate(y_pred):
            seen.setdefault(float(v), i)
        return list(seen.values())[:k]
    order = np.argsort(y_pred)            # low → high predicted grade
    picks = np.linspace(0, n - 1, num=min(k, n)).round().astype(int)
    return [int(order[p]) for p in picks]


def _local_example(i: int, contribs: np.ndarray, X: pd.DataFrame, feat: list[str],
                   y_true, y_pred, pred_class: int | None, top: int) -> dict:
    row = contribs[i] if pred_class is None else contribs[i, pred_class]
    order = np.argsort(np.abs(row))[::-1][:top]
    return {
        "y_true": _num(y_true[i]),
        "y_pred": _num(y_pred[i]),
        "top_features": [
            {"feature": feat[j], "shap": round(float(row[j]), 4),
             "value": _num(X.iloc[i, j])}
            for j in order
        ],
    }


def _num(v) -> float | None:
    v = float(v)
    return None if not np.isfinite(v) else round(v, 4)


def explain_dataset(name: str, bundle: Bundle, cfg: DataConfig,
                    n_examples: int = 5, top: int = 6, sample: int = 3000) -> dict | None:
    params, source = load_tuned_params(name, cfg)
    fitted = fit_head_on_all(bundle, cfg, params)
    if fitted is None:
        return None
    head, X, y = fitted
    spec = bundle.spec
    feat = bundle.feature_cols

    # bound cost: SHAP on a deterministic sample of the rows
    if len(X) > sample:
        Xs = X.sample(sample, random_state=cfg.seed)
        ys = y[X.index.get_indexer(Xs.index)]
    else:
        Xs, ys = X, y
    Xs = Xs.reset_index(drop=True)

    contribs = shaped_contribs(head, Xs, len(feat))
    imp = global_importance(contribs)
    ranking = sorted(({"feature": f, "mean_abs_shap": round(float(v), 4)}
                      for f, v in zip(feat, imp)),
                     key=lambda d: d["mean_abs_shap"], reverse=True)

    # agreement between SHAP ranking and the 2C gain ranking (sanity / robustness)
    gain = np.asarray(head.feature_importances_, dtype=float)
    rho = spearmanr(imp, gain).correlation if np.ptp(gain) > 0 else float("nan")

    y_pred = head.predict(Xs)
    idx = _example_indices(y_pred, spec.task_type, n_examples)
    multiclass = contribs.ndim == 3
    examples = []
    for i in idx:
        pc = int(round(float(y_pred[i]))) if multiclass else None
        if multiclass:
            pc = min(max(pc, 0), contribs.shape[1] - 1)
        examples.append(_local_example(i, contribs, Xs, feat, ys, y_pred, pc, top))

    inv_vocab = {v: k for k, v in bundle.label_vocab.items()}
    return {
        "head_source": source,
        "task_type": spec.task_type,
        "n_explained": int(len(Xs)),
        "n_features": len(feat),
        "shap_vs_gain_spearman": None if rho is None or not np.isfinite(rho) else round(float(rho), 3),
        "global_importance": ranking,
        "label_vocab": inv_vocab or None,
        "examples": examples,
    }
