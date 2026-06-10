"""Phase 2G evaluation protocol for the cross-encoder.

Deliberately mirrors :mod:`asag.models.evaluate` (same ``official_split`` /
``kfold`` / ``per_prompt`` logic, same multi-seed mean±std aggregation, same paired
trivial baseline) so neural and GBM numbers are computed identically and are
directly comparable. The only differences are the data source (the encoder-view
``encoder.parquet`` text, not the feature matrix) and the head (a fine-tuned
transformer, not LightGBM).

For the **headline split** it also caches per-item ``(y_true, y_pred, y_cont)``
under ``reports/phase2g/preds/`` — consumed by the error analysis, the paired
bootstrap, and the GBM+DeBERTa hybrid.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from asag.config import DataConfig
from asag.models.baselines import fit_predict_baseline
from asag.models.evaluate import _agg, _mean_over
from asag.models.metrics import compute_metrics
from asag.models.tasks import TaskSpec, get_spec
from asag.neural.trainer import LabelSpace, fit_predict
from asag.utils.logging import get_logger

log = get_logger()

TEXT_COLS = ["question_id", "question_enc", "reference_answer_enc",
             "student_answer_enc", "score", "label", "split", "fold"]


def load_text_df(name: str, cfg: DataConfig) -> pd.DataFrame | None:
    path = cfg.paths.processed / name / "encoder.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path).reset_index(drop=True)


def _baseline_pred(train_df: pd.DataFrame, ls: LabelSpace, spec: TaskSpec, n_test: int) -> np.ndarray:
    """Trivial baseline in metric space (mirrors the GBM pairing)."""
    if spec.task_type == "classification":
        y_tr = ls.targets(train_df)
        return fit_predict_baseline(y_tr[np.isfinite(y_tr)], n_test, spec.task_type)
    y_tr = pd.to_numeric(train_df["score"], errors="coerce").to_numpy(float)
    return fit_predict_baseline(y_tr[np.isfinite(y_tr)], n_test, spec.task_type)


def _max_len(name: str, cfg: DataConfig) -> int:
    return cfg.neural.max_len_overrides.get(name, cfg.neural.max_len)


def _eval_official(name: str, df: pd.DataFrame, spec: TaskSpec, cfg: DataConfig,
                   tok) -> tuple[dict, dict]:
    ncfg = cfg.neural
    max_len = _max_len(name, cfg)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    dev_df = df[df["split"] == "dev"].reset_index(drop=True)
    evals: dict[str, dict] = {}
    cache: dict[str, dict] = {}

    for split in spec.test_splits:
        test_df = df[df["split"] == split].reset_index(drop=True)
        if test_df.empty or train_df.empty:
            evals[split] = {"n_eval": int(len(test_df)), "neural": {}, "baseline": {}}
            continue

        per_seed_n, per_seed_b = [], []
        last_items: dict = {}
        for seed in ncfg.seeds:
            if spec.per_prompt:
                prompts = sorted(test_df["question_id"].astype(str).unique())
                nm, bm = [], []
                yt_all, yp_all, qc_all = [], [], []
                for p in prompts:
                    tr = train_df[train_df["question_id"].astype(str) == p].reset_index(drop=True)
                    te = test_df[test_df["question_id"].astype(str) == p].reset_index(drop=True)
                    dv = dev_df[dev_df["question_id"].astype(str) == p].reset_index(drop=True)
                    if tr.empty or te.empty:
                        continue
                    r = fit_predict(tr, te, spec, ncfg, seed, max_len=max_len,
                                    dev_df=dv if len(dv) else None, tokenizer=tok)
                    ls = LabelSpace(spec, tr)
                    base = _baseline_pred(tr, ls, spec, len(te))
                    nm.append(compute_metrics(r["y_true"], r["y_pred"], spec.metrics))
                    bm.append(compute_metrics(r["y_true"], base, spec.metrics))
                    yt_all.append(r["y_true"]); yp_all.append(r["y_pred"]); qc_all.append(np.full(len(te), p, dtype=object))
                per_seed_n.append(_mean_over(nm)); per_seed_b.append(_mean_over(bm))
                last_items = {"y_true": np.concatenate(yt_all), "y_pred": np.concatenate(yp_all),
                              "group": np.concatenate(qc_all)}
            else:
                r = fit_predict(train_df, test_df, spec, ncfg, seed, max_len=max_len,
                                dev_df=dev_df if len(dev_df) else None, tokenizer=tok)
                ls = LabelSpace(spec, train_df)
                base = _baseline_pred(train_df, ls, spec, len(test_df))
                per_seed_n.append(compute_metrics(r["y_true"], r["y_pred"], spec.metrics))
                per_seed_b.append(compute_metrics(r["y_true"], base, spec.metrics))
                last_items = {"y_true": r["y_true"], "y_pred": r["y_pred"], "y_cont": r["y_cont"],
                              "group": np.zeros(len(test_df))}
            log.info(f"  {name}/{split} seed={seed} "
                     f"{spec.headline}={per_seed_n[-1].get(spec.headline, float('nan')):.4f}")
        evals[split] = {"n_eval": int(len(test_df)),
                        "neural": _agg(per_seed_n), "baseline": _agg(per_seed_b)}
        cache[split] = last_items
    return evals, cache


def _eval_kfold(name: str, df: pd.DataFrame, spec: TaskSpec, cfg: DataConfig,
                tok) -> tuple[dict, dict]:
    ncfg = cfg.neural
    max_len = _max_len(name, cfg)
    folds = sorted(int(f) for f in df["fold"].unique() if int(f) >= 0)
    if not folds:
        return {"cv": {"n_eval": 0, "neural": {}, "baseline": {}}}, {}

    per_seed_n, per_seed_b = [], []
    last_items: dict = {}
    for seed in ncfg.seeds:
        yt, yp, yb, yc, grp = [], [], [], [], []
        for f in folds:
            tr = df[(df["fold"] != f) & (df["fold"] >= 0)].reset_index(drop=True)
            te = df[df["fold"] == f].reset_index(drop=True)
            if tr.empty or te.empty:
                continue
            r = fit_predict(tr, te, spec, ncfg, seed, max_len=max_len, tokenizer=tok)
            ls = LabelSpace(spec, tr)
            base = _baseline_pred(tr, ls, spec, len(te))
            yt.append(r["y_true"]); yp.append(r["y_pred"]); yb.append(base)
            yc.append(r["y_cont"]); grp.append(np.full(len(te), f))
        if not yt:
            continue
        YT = np.concatenate(yt)
        per_seed_n.append(compute_metrics(YT, np.concatenate(yp), spec.metrics))
        per_seed_b.append(compute_metrics(YT, np.concatenate(yb), spec.metrics))
        last_items = {"y_true": YT, "y_pred": np.concatenate(yp),
                      "y_cont": np.concatenate(yc), "group": np.concatenate(grp)}
        log.info(f"  {name}/cv seed={seed} "
                 f"{spec.headline}={per_seed_n[-1].get(spec.headline, float('nan')):.4f}")
    return ({"cv": {"n_eval": int(sum(df["fold"] >= 0)), "n_folds": len(folds),
                    "neural": _agg(per_seed_n), "baseline": _agg(per_seed_b)}},
            {"cv": last_items})


def evaluate_neural(name: str, cfg: DataConfig, tok) -> tuple[dict, dict] | None:
    spec = get_spec(name)
    df = load_text_df(name, cfg)
    if df is None:
        log.warning(f"{name}: encoder.parquet missing — run `make preprocess`; skipping")
        return None
    evals, cache = (_eval_official(name, df, spec, cfg, tok) if spec.protocol == "official_split"
                    else _eval_kfold(name, df, spec, cfg, tok))
    headline_split = spec.test_splits[-1] if spec.protocol == "official_split" else "cv"
    h = evals.get(headline_split, {})
    result = {
        "task_type": spec.task_type, "protocol": spec.protocol, "target": spec.target,
        "per_prompt": spec.per_prompt, "backbone": cfg.neural.backbone,
        "seeds": list(cfg.neural.seeds), "evaluations": evals,
        "headline": {"split": headline_split, "metric": spec.headline,
                     "neural": h.get("neural", {}).get(spec.headline, {}),
                     "baseline": h.get("baseline", {}).get(spec.headline, {})},
    }
    g = result["headline"]["neural"].get("mean", float("nan"))
    log.info(f"{name}: {spec.headline}@{headline_split} neural={g:.4f}")
    return result, cache
