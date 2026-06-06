"""Phase 2D — Optuna hyperparameter optimization for the LightGBM fusion head.

The search is **per dataset** and its objective is computed on training-side data
only — the held-out test splits are never touched during tuning:

* If the dataset ships an official ``dev`` split (ASAP-SAS does; SemEval/SAF too
  when present), the objective fits on ``train`` and scores the headline metric
  on ``dev``.
* Otherwise (the k-fold datasets — Mohler / Powergrading / MIND-CA, or any
  official-split dataset without a dev split), the objective is the mean headline
  over a fresh inner ``StratifiedKFold`` carved from the training rows.

This is the *inner-CV / dev* strategy: fast and reviewer-defensible, with a
documented, mild HPO optimism for the k-fold case (full nested CV is deferred).
The tuned :class:`LightGBMCfg` is then handed to the unchanged Phase 2C protocol
(``evaluate.evaluate_dataset(..., head_params=tuned)``) for the final multi-seed
numbers — so HPO never re-implements the official_split / kfold / per_prompt logic.

``optuna`` is imported lazily (mirrors ``LIGHTGBM_AVAILABLE`` in ``fusion``); the
module imports without it and tests ``importorskip`` it.
"""

from __future__ import annotations

import numpy as np

try:  # optional dependency — see pyproject Phase 2D block
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover
    optuna = None
    OPTUNA_AVAILABLE = False

from asag.config import DataConfig, LightGBMCfg
from asag.data.splits import make_stratified_kfold
from asag.models.data import Bundle, make_y
from asag.models.evaluate import fit_predict_arrays
from asag.models.metrics import compute_metrics
from asag.utils.logging import get_logger

log = get_logger()

# Documented search space (also serialized into hpo.json for the paper appendix).
SEARCH_SPACE: dict[str, dict] = {
    "learning_rate": {"type": "float", "low": 5e-3, "high": 0.3, "log": True},
    "num_leaves": {"type": "int", "low": 15, "high": 255},
    "n_estimators": {"type": "int", "low": 100, "high": 800, "step": 50},
    "min_child_samples": {"type": "int", "low": 5, "high": 80},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
    "reg_alpha": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
}

# Headline metrics where a lower value is better (none in the current registry,
# but the direction logic stays correct if rmse/mae ever become a headline).
_LOWER_IS_BETTER = {"rmse", "mae"}


def suggest_params(trial, base: LightGBMCfg) -> LightGBMCfg:
    """Sample a :class:`LightGBMCfg` from ``trial`` over :data:`SEARCH_SPACE`."""
    p = dict(
        learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )
    return LightGBMCfg(**{**base.model_dump(), **p})


def _finite(df, bundle: Bundle):
    """Rows whose modelling target is finite (drops unlabelled rows)."""
    return df[np.isfinite(make_y(df, bundle))]


def _train_pool(bundle: Bundle):
    """Training-side rows the objective is allowed to see (never the test splits)."""
    df = bundle.df
    if bundle.spec.protocol == "official_split":
        pool = df[df["split"] == "train"]
    else:  # kfold — all labelled rows participate (final eval is OOF over folds)
        pool = df[df["fold"] >= 0]
    return _finite(pool, bundle)


def _dev_df(bundle: Bundle):
    """The official ``dev`` split if the dataset has one, else None."""
    df = bundle.df
    if "dev" not in set(df["split"].unique()):
        return None
    dev = _finite(df[df["split"] == "dev"], bundle)
    return dev if not dev.empty else None


def _headline(y_true, y_pred, bundle: Bundle) -> float:
    return compute_metrics(y_true, y_pred, (bundle.spec.headline,))[bundle.spec.headline]


def _safe_mean(scores: list[float]) -> float:
    """Mean of the finite scores, or nan — avoids numpy's empty-slice warning."""
    finite = [s for s in scores if np.isfinite(s)]
    return float(np.mean(finite)) if finite else float("nan")


def _score_on_dev(params: LightGBMCfg, train_df, dev_df, bundle: Bundle,
                  cfg: DataConfig, seed: int) -> float:
    spec = bundle.spec
    if spec.per_prompt:
        scores = []
        for p in sorted(dev_df["question_id"].astype(str).unique()):
            tr = train_df[train_df["question_id"].astype(str) == p]
            de = dev_df[dev_df["question_id"].astype(str) == p]
            if tr.empty or de.empty:
                continue
            y_te, gbm_pred, _ = fit_predict_arrays(tr, de, bundle, cfg, seed, params)
            scores.append(_headline(y_te, gbm_pred, bundle))
        return _safe_mean(scores)
    y_te, gbm_pred, _ = fit_predict_arrays(train_df, dev_df, bundle, cfg, seed, params)
    return _headline(y_te, gbm_pred, bundle)


def _score_inner_cv(params: LightGBMCfg, pool, bundle: Bundle,
                    cfg: DataConfig, seed: int, k: int) -> float:
    spec = bundle.spec
    pool = pool.reset_index(drop=True)
    stratify_on = "label" if spec.task_type == "classification" else "score"
    folds = make_stratified_kfold(pool, k=k, seed=seed, stratify_on=stratify_on)
    scores = []
    for f in sorted(folds.unique()):
        te = pool[folds == f]
        tr = pool[folds != f]
        if tr.empty or te.empty:
            continue
        y_te, gbm_pred, _ = fit_predict_arrays(tr, te, bundle, cfg, seed, params)
        scores.append(_headline(y_te, gbm_pred, bundle))
    return _safe_mean(scores)


def tune_dataset(bundle: Bundle, cfg: DataConfig) -> tuple[LightGBMCfg, dict]:
    """Run Optuna for one dataset; return ``(tuned_cfg, study_summary)``.

    Falls back to the config defaults (and a summary marked ``skipped``) when HPO
    is disabled or Optuna is unavailable, so the caller can always proceed.
    """
    spec = bundle.spec
    hpo = cfg.model.hpo
    base = cfg.model.lightgbm
    if not hpo.enabled or not OPTUNA_AVAILABLE:
        reason = "hpo.enabled is false" if not hpo.enabled else "optuna not installed"
        log.warning(f"{bundle.name}: HPO skipped ({reason}); using config defaults")
        return base, {"status": "skipped", "reason": reason}

    pool = _train_pool(bundle)
    dev = _dev_df(bundle)
    method = "dev" if dev is not None else "inner_cv"
    maximize = spec.headline not in _LOWER_IS_BETTER
    worst = -1e9 if maximize else 1e9

    def objective(trial) -> float:
        params = suggest_params(trial, base)
        if dev is not None:
            val = _score_on_dev(params, pool, dev, bundle, cfg, hpo.seed)
        else:
            val = _score_inner_cv(params, pool, bundle, cfg, hpo.seed, hpo.inner_folds)
        return float(val) if np.isfinite(val) else worst

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=hpo.seed)
    study = optuna.create_study(direction="maximize" if maximize else "minimize",
                                sampler=sampler)
    study.optimize(objective, n_trials=hpo.n_trials, timeout=hpo.timeout_s,
                   show_progress_bar=False)

    tuned = LightGBMCfg(**{**base.model_dump(), **study.best_params})
    summary = {
        "status": "ok",
        "validation": method,          # "dev" | "inner_cv"
        "inner_folds": hpo.inner_folds if method == "inner_cv" else None,
        "headline_metric": spec.headline,
        "direction": "maximize" if maximize else "minimize",
        "n_trials": len(study.trials),
        "best_value": float(study.best_value),
        "best_params": tuned.model_dump(),
        "search_space": SEARCH_SPACE,
        "n_train_pool": int(len(pool)),
        "n_dev": int(len(dev)) if dev is not None else 0,
    }
    log.info(f"{bundle.name}: HPO best {spec.headline}={study.best_value:.4f} "
             f"({method}, {len(study.trials)} trials)")
    return tuned, summary
