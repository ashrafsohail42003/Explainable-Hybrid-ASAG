"""Phase 2C evaluation protocol.

Two protocols, both averaged over the config seeds (mean ± std):

* ``official_split`` — fit on the ``train`` split, score each held-out test split
  (SemEval reports all three: test_ua / test_uq / **test_ud** cross-domain). For
  ASAP-SAS, train/score **one model per prompt** and average QWK across prompts.
* ``kfold`` — rotate the materialized ``fold`` column (built in Phase 1); metrics
  are computed on the pooled out-of-fold predictions, once per seed.

Every learned result is paired with a trivial baseline (majority/mean/median) so
the report can show the ΔQWK the head actually buys.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from asag.config import DataConfig, LightGBMCfg, ensure_dirs, load_data_config
from asag.data.splits import make_grouped_kfold, make_stratified_kfold
from asag.models import MODELS_SCHEMA_VERSION
from asag.models.baselines import fit_predict_baseline, question_shortcut_predict
from asag.models.data import Bundle, load_bundle, make_X, make_y, question_prior
from asag.models.fusion import LIGHTGBM_AVAILABLE, LgbmFusionHead
from asag.models.metrics import compute_metrics
from asag.models.tasks import REGISTRY, TaskSpec, get_spec
from asag.utils.logging import get_logger
from asag.utils.seed import set_global_seed

log = get_logger()


def _agg(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate per-seed metric dicts into {metric: {mean, std, n}}."""
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([d.get(k, np.nan) for d in metric_dicts], dtype=float)
        finite = vals[np.isfinite(vals)]
        out[k] = {
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "std": float(np.std(finite)) if finite.size else float("nan"),
            "n": int(finite.size),
        }
    return out


def _mean_over(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across a list (e.g. across prompts), ignoring nan."""
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {
        k: float(np.nanmean([d.get(k, np.nan) for d in metric_dicts]))
        for k in keys
    }


def fit_predict_arrays(
    train_df: pd.DataFrame, test_df: pd.DataFrame, bundle: Bundle,
    cfg: DataConfig, seed: int, head_params: LightGBMCfg | None = None,
    with_qprior: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit head + baseline, return per-item ``(y_true, gbm_pred, base_pred)``.

    ``head_params`` overrides ``cfg.model.lightgbm`` (Phase 2D passes the tuned
    config; ``None`` reproduces Phase 2C). On a degenerate train split (no rows,
    or a single class) the GBM falls back to the baseline prediction so callers
    always get three aligned arrays — the significance module relies on this.

    ``with_qprior`` appends the fold-safe question-difficulty prior
    (:func:`question_prior`) as an extra feature; defaults to
    ``cfg.model.qprior_enabled``. It is NaN for unseen questions, so it only ever
    adds signal where test questions are also in train (the seen-question /
    ``test_ua`` upper bound) — under the honest grouped/unseen protocol it is
    NaN at test and contributes nothing by construction.
    """
    spec = bundle.spec
    params = head_params or cfg.model.lightgbm
    if with_qprior is None:
        with_qprior = getattr(cfg.model, "qprior_enabled", False)
    set_global_seed(seed)
    X_tr, y_tr = make_X(train_df, bundle.feature_cols), make_y(train_df, bundle)
    X_te, y_te = make_X(test_df, bundle.feature_cols), make_y(test_df, bundle)

    if with_qprior and spec.task_type != "classification":
        tr_prior, te_prior = question_prior(train_df, test_df, spec)
        X_tr = X_tr.assign(qprior_train_mean=tr_prior)
        X_te = X_te.assign(qprior_train_mean=te_prior)

    m = np.isfinite(y_tr)
    X_tr, y_tr = X_tr.loc[m], y_tr[m]

    base_pred = fit_predict_baseline(y_tr, len(y_te), spec.task_type)
    if y_tr.size == 0 or (spec.task_type == "classification" and np.unique(y_tr).size < 2):
        return y_te, base_pred, base_pred

    head = LgbmFusionHead(spec.task_type, params, seed).fit(X_tr, y_tr)
    gbm_pred = head.predict(X_te)
    return y_te, gbm_pred, base_pred


def _shortcut_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      bundle: Bundle) -> dict[str, float]:
    """Per-question memorization-control metrics on one train/test split."""
    spec = bundle.spec
    y_tr = make_y(train_df, bundle)
    y_te = make_y(test_df, bundle)
    pred = question_shortcut_predict(
        y_tr, train_df["question_id"], test_df["question_id"], spec.task_type)
    return compute_metrics(y_te, pred, spec.metrics)


def _fit_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, bundle: Bundle,
              cfg: DataConfig, seed: int,
              head_params: LightGBMCfg | None = None) -> tuple[dict, dict]:
    """Fit the head on train_df, score test_df; return (gbm, baseline) metrics."""
    spec = bundle.spec
    y_te, gbm_pred, base_pred = fit_predict_arrays(
        train_df, test_df, bundle, cfg, seed, head_params)
    return (compute_metrics(y_te, gbm_pred, spec.metrics),
            compute_metrics(y_te, base_pred, spec.metrics))


def _eval_official(bundle: Bundle, cfg: DataConfig,
                   head_params: LightGBMCfg | None = None) -> dict:
    spec, df = bundle.spec, bundle.df
    y_all = make_y(df, bundle)
    df = df.assign(_y=y_all)
    train_df = df[(df["split"] == "train") & np.isfinite(df["_y"])]
    seeds = cfg.model.seeds
    evals: dict[str, dict] = {}

    for split in spec.test_splits:
        test_df = df[(df["split"] == split) & np.isfinite(df["_y"])]
        if test_df.empty or train_df.empty:
            evals[split] = {"n_eval": int(len(test_df)), "gbm": {}, "baseline": {}}
            continue

        if spec.per_prompt:
            prompts = sorted(test_df["question_id"].astype(str).unique())
            per_seed_gbm, per_seed_base = [], []
            shortcut_p = []
            prompt_acc: dict[str, list[dict]] = defaultdict(list)
            for seed in seeds:
                gbm_p, base_p = [], []
                for p in prompts:
                    tr = train_df[train_df["question_id"].astype(str) == p]
                    te = test_df[test_df["question_id"].astype(str) == p]
                    if tr.empty or te.empty:
                        continue
                    gm, bm = _fit_eval(tr, te, bundle, cfg, seed, head_params)
                    gbm_p.append(gm); base_p.append(bm); prompt_acc[p].append(gm)
                per_seed_gbm.append(_mean_over(gbm_p))
                per_seed_base.append(_mean_over(base_p))
            # shortcut is seed-independent (deterministic per-question stat)
            shortcut_p = [_shortcut_metrics(
                train_df[train_df["question_id"].astype(str) == p],
                test_df[test_df["question_id"].astype(str) == p], bundle)
                for p in prompts
                if not train_df[train_df["question_id"].astype(str) == p].empty
                and not test_df[test_df["question_id"].astype(str) == p].empty]
            evals[split] = {
                "n_eval": int(len(test_df)),
                "n_prompts": len(prompts),
                "gbm": _agg(per_seed_gbm),
                "baseline": _agg(per_seed_base),
                "question_shortcut": _agg([_mean_over(shortcut_p)]) if shortcut_p else {},
                "per_prompt": {p: _agg(v) for p, v in prompt_acc.items()},
            }
        else:
            per_seed_gbm, per_seed_base = [], []
            for seed in seeds:
                gm, bm = _fit_eval(train_df, test_df, bundle, cfg, seed, head_params)
                per_seed_gbm.append(gm); per_seed_base.append(bm)
            evals[split] = {
                "n_eval": int(len(test_df)),
                "gbm": _agg(per_seed_gbm),
                "baseline": _agg(per_seed_base),
                "question_shortcut": _agg([_shortcut_metrics(train_df, test_df, bundle)]),
            }
    return evals


def _pooled_oof(df: pd.DataFrame, fold: np.ndarray, bundle: Bundle,
                cfg: DataConfig, head_params: LightGBMCfg | None,
                finite: np.ndarray) -> tuple[list[dict], list[dict], list[dict]]:
    """Pooled out-of-fold metrics over a fold assignment, per seed.

    Returns ``(gbm, baseline, shortcut)`` per-seed metric-dict lists. The
    shortcut is the per-question memorization control (see :func:`_shortcut_metrics`).
    """
    spec = bundle.spec
    folds = sorted(int(f) for f in np.unique(fold) if int(f) >= 0)
    g_seed, b_seed, s_seed = [], [], []
    for seed in cfg.model.seeds:
        yt, gp, bp, sc_t, sc_p = [], [], [], [], []
        for f in folds:
            te_m = (fold == f) & finite
            tr_m = (fold != f) & (fold >= 0) & finite
            tr, te = df[tr_m], df[te_m]
            if tr.empty or te.empty:
                continue
            y_te, gbm_pred, base_pred = fit_predict_arrays(tr, te, bundle, cfg, seed, head_params)
            yt.append(y_te); gp.append(gbm_pred); bp.append(base_pred)
            sc_t.append(make_y(te, bundle))
            sc_p.append(question_shortcut_predict(
                make_y(tr, bundle), tr["question_id"], te["question_id"], spec.task_type))
        if not yt:
            continue
        YT = np.concatenate(yt)
        g_seed.append(compute_metrics(YT, np.concatenate(gp), spec.metrics))
        b_seed.append(compute_metrics(YT, np.concatenate(bp), spec.metrics))
        s_seed.append(compute_metrics(np.concatenate(sc_t), np.concatenate(sc_p), spec.metrics))
    return g_seed, b_seed, s_seed


def _eval_kfold(bundle: Bundle, cfg: DataConfig,
                head_params: LightGBMCfg | None = None) -> dict:
    """Primary = grouped (materialized ``fold``, unseen questions). Also reports a
    transient *seen-question upper bound* (legacy score-stratified folds) and the
    generalization gap, so the question-memorization shortcut is quantified."""
    spec, df = bundle.spec, bundle.df
    y_all = make_y(df, bundle)
    finite = np.isfinite(y_all)
    grouped_fold = df["fold"].to_numpy()
    folds = sorted(int(f) for f in np.unique(grouped_fold) if int(f) >= 0)
    if not folds:
        log.warning(f"{bundle.name}: no k-fold folds (fold column all -1); skipping kfold eval")
        return {"cv": {"n_eval": 0, "gbm": {}, "baseline": {}}}

    n_eval = int((finite & (df["fold"] >= 0)).sum())
    g, b, s = _pooled_oof(df, grouped_fold, bundle, cfg, head_params, finite)

    # transient seen-question upper bound: legacy score-stratified folds (leak qid)
    strat_fold = make_stratified_kfold(
        df, k=cfg.splits.cv_k_folds, seed=cfg.seed, stratify_on=cfg.splits.stratify_on).to_numpy()
    ub_g, _, ub_s = _pooled_oof(df, strat_fold, bundle, cfg, head_params, finite)

    grouped_gbm, upper_gbm = _agg(g), _agg(ub_g)
    gap = (upper_gbm.get(spec.headline, {}).get("mean", float("nan"))
           - grouped_gbm.get(spec.headline, {}).get("mean", float("nan")))
    return {"cv": {
        "n_eval": n_eval,
        "n_folds": len(folds),
        "cv_strategy": "grouped_by_question",
        "gbm": grouped_gbm,
        "baseline": _agg(b),
        "question_shortcut": _agg(s),
        "upper_bound_seen_question": upper_gbm,
        "upper_bound_question_shortcut": _agg(ub_s),
        "generalization_gap": round(float(gap), 4) if np.isfinite(gap) else None,
    }}


def _feature_importance(bundle: Bundle, cfg: DataConfig) -> dict[str, float]:
    """Gain importances from one head trained on all valid rows (illustrative)."""
    spec, df = bundle.spec, bundle.df
    y = make_y(df, bundle)
    m = np.isfinite(y)
    X, y = make_X(df[m], bundle.feature_cols), y[m]
    if y.size == 0 or (spec.task_type == "classification" and np.unique(y).size < 2):
        return {}
    set_global_seed(cfg.seed)
    head = LgbmFusionHead(spec.task_type, cfg.model.lightgbm, cfg.seed).fit(X, y)
    imp = {c: float(v) for c, v in zip(bundle.feature_cols, head.feature_importances_)}
    return dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))


def evaluate_dataset(name: str, cfg: DataConfig,
                     head_params: LightGBMCfg | None = None) -> dict | None:
    spec = get_spec(name)
    bundle = load_bundle(name, cfg, spec)
    if bundle is None:
        log.warning(f"{name}: features.parquet missing — run `make features`; skipping")
        return None

    evals = (_eval_official(bundle, cfg, head_params) if spec.protocol == "official_split"
             else _eval_kfold(bundle, cfg, head_params))

    # headline split: cross-domain / unseen-question for official; grouped CV pool
    # for kfold. For both, the headline is the *honest* (hardest) generalization.
    headline_split = spec.test_splits[-1] if spec.protocol == "official_split" else "cv"
    h = evals.get(headline_split, {})

    # generalization gap: seen-question upper bound minus the honest headline.
    if spec.protocol == "kfold":
        gen_gap = h.get("generalization_gap")
    else:
        gen_gap = None
        if "test_ua" in evals and headline_split != "test_ua":
            ua = evals["test_ua"].get("gbm", {}).get(spec.headline, {}).get("mean", float("nan"))
            hd = h.get("gbm", {}).get(spec.headline, {}).get("mean", float("nan"))
            if np.isfinite(ua) and np.isfinite(hd):
                gen_gap = round(float(ua - hd), 4)

    headline = {
        "split": headline_split,
        "metric": spec.headline,
        "gbm": h.get("gbm", {}).get(spec.headline, {}),
        "baseline": h.get("baseline", {}).get(spec.headline, {}),
        "question_shortcut": h.get("question_shortcut", {}).get(spec.headline, {}),
        "generalization_gap": gen_gap,
    }
    result = {
        "task_type": spec.task_type,
        "protocol": spec.protocol,
        "target": spec.target,
        "per_prompt": spec.per_prompt,
        "n_rows": int(len(bundle.df)),
        "n_features": len(bundle.feature_cols),
        "seeds": list(cfg.model.seeds),
        "evaluations": evals,
        "headline": headline,
    }
    g = headline["gbm"].get("mean", float("nan"))
    b = headline["baseline"].get("mean", float("nan"))
    log.info(f"{name}: {spec.headline}@{headline_split} gbm={g:.4f} baseline={b:.4f}")
    return result


# ----------------------------- reporting ---------------------------------

def _flatten_rows(name: str, result: dict) -> list[dict]:
    rows: list[dict] = []
    for split, ev in result["evaluations"].items():
        for model in ("gbm", "baseline", "question_shortcut", "upper_bound_seen_question"):
            for metric, stats in ev.get(model, {}).items():
                rows.append({
                    "dataset": name,
                    "task_type": result["task_type"],
                    "protocol": result["protocol"],
                    "split": split,
                    "model": model,
                    "metric": metric,
                    "mean": round(stats.get("mean", float("nan")), 4),
                    "std": round(stats.get("std", float("nan")), 4),
                })
    return rows


def _write_results(cfg: DataConfig, results: dict[str, dict],
                   importances: dict[str, dict]) -> None:
    out_dir = cfg.paths.reports / "phase2c"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = {"schema_version": MODELS_SCHEMA_VERSION,
           "fusion_head": cfg.model.fusion_head,
           "seeds": list(cfg.model.seeds),
           "lightgbm": cfg.model.lightgbm.model_dump(),
           "datasets": results}
    (out_dir / "results.json").write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")

    rows = [r for name, res in results.items() for r in _flatten_rows(name, res)]
    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    (out_dir / "feature_importance.json").write_text(
        json.dumps(importances, indent=2), encoding="utf-8")
    log.info(f"wrote {out_dir / 'results.json'}, results.csv, feature_importance.json")


def _write_figures(cfg: DataConfig, results: dict[str, dict],
                   importances: dict[str, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = cfg.paths.figures
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) headline metric per dataset: GBM vs baseline, with seed std error bars.
    names = list(results.keys())
    g_mean = [results[n]["headline"]["gbm"].get("mean", np.nan) for n in names]
    g_std = [results[n]["headline"]["gbm"].get("std", 0.0) or 0.0 for n in names]
    b_mean = [results[n]["headline"]["baseline"].get("mean", np.nan) for n in names]
    labels = [f"{n}\n({results[n]['headline']['metric']}@{results[n]['headline']['split']})" for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(names)), 5))
    ax.bar(x - w / 2, g_mean, w, yerr=g_std, capsize=4, label="GBM fusion head", color="#2c7fb8")
    ax.bar(x + w / 2, b_mean, w, label="naive baseline", color="#bdbdbd")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("headline metric"); ax.set_title("Phase 2C — late-fusion GBM head vs baseline")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "phase2c_results.png", dpi=120); plt.close(fig)

    # 2) top feature importances (small multiples over datasets that have them).
    have = [(n, importances[n]) for n in names if importances.get(n)]
    if have:
        cols = min(3, len(have)); rows = int(np.ceil(len(have) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.2 * rows), squeeze=False)
        for i, (n, imp) in enumerate(have):
            ax = axes[i // cols][i % cols]
            top = list(imp.items())[:10][::-1]
            ax.barh([k for k, _ in top], [v for _, v in top], color="#41ab5d")
            ax.set_title(n, fontsize=9); ax.tick_params(labelsize=7)
        for j in range(len(have), rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.suptitle("Phase 2C — LightGBM gain importance (top 10)", fontsize=11)
        fig.tight_layout(); fig.savefig(fig_dir / "phase2c_feature_importance.png", dpi=120); plt.close(fig)
    log.info(f"wrote figures to {fig_dir}")


def run_all(cfg: DataConfig | None = None, only: list[str] | None = None) -> dict[str, dict]:
    cfg = cfg or load_data_config()
    ensure_dirs(cfg)
    set_global_seed(cfg.seed)
    if not cfg.model.enabled:
        log.warning("model.enabled is false — nothing to do")
        return {}
    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError("lightgbm is not installed; run `uv pip install lightgbm`")

    names = only or [n for n in REGISTRY if (cfg.paths.processed / n / "features.parquet").exists()]
    results: dict[str, dict] = {}
    importances: dict[str, dict] = {}
    for name in names:
        res = evaluate_dataset(name, cfg)
        if res is None:
            continue
        results[name] = res
        bundle = load_bundle(name, cfg, get_spec(name))
        importances[name] = _feature_importance(bundle, cfg) if bundle else {}

    if results:
        _write_results(cfg, results, importances)
        _write_figures(cfg, results, importances)
    return results


if __name__ == "__main__":
    import sys

    run_all(only=sys.argv[1:] or None)
