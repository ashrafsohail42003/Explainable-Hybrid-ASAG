"""Shared helpers for the Phase 2F XAI layer."""

from __future__ import annotations

import json

import numpy as np

from asag.config import DataConfig, LightGBMCfg
from asag.models.data import Bundle, make_X, make_y
from asag.models.fusion import LgbmFusionHead


def load_tuned_params(name: str, cfg: DataConfig) -> tuple[LightGBMCfg, str]:
    """Phase 2D tuned params for ``name`` if present, else the config defaults.

    Returns ``(params, source)`` where source is ``"phase2d_tuned"`` or ``"config_default"``
    so the report can state which head was explained.
    """
    path = cfg.paths.reports / "phase2d" / "results.json"
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            tuned = doc.get("datasets", {}).get(name, {}).get("lightgbm_tuned")
            if tuned:
                return LightGBMCfg(**tuned), "phase2d_tuned"
        except (json.JSONDecodeError, OSError):
            pass
    return cfg.model.lightgbm, "config_default"


def fit_head_on_all(bundle: Bundle, cfg: DataConfig, params: LightGBMCfg):
    """Train one head on every valid row (illustrative, like the 2C importance head).

    Returns ``(head, X, y)`` or ``None`` if the target is degenerate.
    """
    spec, df = bundle.spec, bundle.df
    y = make_y(df, bundle)
    m = np.isfinite(y)
    X, y = make_X(df[m], bundle.feature_cols), y[m]
    if y.size == 0 or (spec.task_type == "classification" and np.unique(y).size < 2):
        return None
    head = LgbmFusionHead(spec.task_type, params, cfg.seed).fit(X, y)
    return head, X, y


def shaped_contribs(head: LgbmFusionHead, X, n_features: int) -> np.ndarray:
    """Per-feature TreeSHAP contributions, base column dropped.

    Shape ``(n, n_features)`` for regression/ordinal/binary, or ``(n, n_classes,
    n_features)`` for multiclass — the XAI callers handle both.
    """
    arr, k = head.pred_contrib(X)
    n, f = arr.shape[0], n_features
    if k <= 1:
        return arr[:, :f]
    return arr.reshape(n, k, f + 1)[:, :, :f]


def global_importance(contribs: np.ndarray) -> np.ndarray:
    """Mean |SHAP| per feature, averaged over samples (and classes if multiclass)."""
    ab = np.abs(contribs)
    return ab.mean(axis=tuple(range(ab.ndim - 1)))
