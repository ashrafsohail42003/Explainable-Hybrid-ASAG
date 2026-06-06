"""The late-fusion head: a NaN-native gradient-boosted tree (LightGBM).

One class serves all three task types. Classification uses ``LGBMClassifier``
(returns class codes). Ordinal and regression use ``LGBMRegressor``; the ordinal
variant rounds-and-clips its continuous output to the integer grade range seen in
training (``regression_threshold`` strategy — QWK-friendly and robust to skew).

LightGBM is imported lazily so the module imports even when the wheel is absent
(tests ``importorskip`` it); ``LIGHTGBM_AVAILABLE`` lets callers fail gracefully.
"""

from __future__ import annotations

import numpy as np

try:  # optional dependency — see pyproject Phase 2C block
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    lgb = None
    LIGHTGBM_AVAILABLE = False

from asag.config import LightGBMCfg


class LgbmFusionHead:
    def __init__(self, task_type: str, params: LightGBMCfg, seed: int):
        if not LIGHTGBM_AVAILABLE:  # pragma: no cover
            raise RuntimeError("lightgbm is not installed; `uv pip install lightgbm`")
        self.task_type = task_type
        self.seed = seed
        self._lo: float | None = None
        self._hi: float | None = None
        common = dict(
            n_estimators=params.n_estimators,
            learning_rate=params.learning_rate,
            num_leaves=params.num_leaves,
            min_child_samples=params.min_child_samples,
            subsample=params.subsample,
            colsample_bytree=params.colsample_bytree,
            reg_alpha=params.reg_alpha,
            reg_lambda=params.reg_lambda,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        # Row bagging only engages when sampled every iteration; without this the
        # ``subsample`` value is silently ignored and the head stays deterministic.
        if params.subsample < 1.0:
            common["subsample_freq"] = 1
        if task_type == "classification":
            self.model = lgb.LGBMClassifier(**common)
        else:
            self.model = lgb.LGBMRegressor(**common)

    def fit(self, X, y) -> "LgbmFusionHead":
        y = np.asarray(y, dtype=float)
        if self.task_type == "classification":
            self.model.fit(X, y.astype(int))
        else:
            if self.task_type == "ordinal":
                self._lo = float(np.min(y))
                self._hi = float(np.max(y))
            self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        raw = np.asarray(self.model.predict(X), dtype=float)
        if self.task_type == "ordinal":
            raw = np.rint(raw)
            if self._lo is not None:
                raw = np.clip(raw, self._lo, self._hi)
        elif self.task_type == "classification":
            raw = np.rint(raw)
        return raw

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(self.model.feature_importances_, dtype=float)

    def pred_contrib(self, X) -> tuple[np.ndarray, int]:
        """Exact per-feature TreeSHAP contributions via LightGBM's ``pred_contrib``.

        Avoids the heavy ``shap`` dependency (TreeSHAP is built into LightGBM).
        Returns ``(arr, n_blocks)`` with the raw LightGBM layout — the last column
        of each block is that block's base/expected value:

        * regression / ordinal / **binary** → ``arr`` is ``(n, n_features + 1)``, ``n_blocks = 1``
        * **multiclass** classification      → ``arr`` is ``(n, n_classes * (n_features + 1))``

        ``n_blocks`` is derived from the array width (binary returns a single block
        even though ``n_classes_ == 2``). The Phase 2F XAI layer reshapes/aggregates
        this; ordinal heads are explained on the raw output (before round-and-clip).
        """
        arr = np.asarray(self.model.predict(X, pred_contrib=True), dtype=float)
        n_blocks = arr.shape[1] // (int(self.model.n_features_in_) + 1)
        return arr, n_blocks
