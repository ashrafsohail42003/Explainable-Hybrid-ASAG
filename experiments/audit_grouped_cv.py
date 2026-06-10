"""Leakage / memorization audit — standalone, non-destructive.

Reads the existing ``features.parquet`` (no re-preprocessing, no SBERT) and, for
each k-fold dataset, compares the *legacy* stratified-by-score folds (the
materialized ``fold`` column) against *grouped* folds that hold out whole
``question_id`` groups (StratifiedGroupKFold). For each it scores:

  * the GBM late-fusion head (headline metric, pooled OOF), and
  * a per-question shortcut predictor (question-mean for regression/ordinal,
    per-question majority for classification) — this is the thing that should
    work under stratified folds (question seen in train) and DIE under grouped
    folds (question unseen).

The gap between the two strategies is the inflation. Writes
``reports/phase4_audit/grouped_cv_audit.json``.

Run (Windows venv, repo root):
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
    "C:/Users/MSI/.cache/asag-venvs/asag-py311/Scripts/python.exe" \
    experiments/audit_grouped_cv.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from asag.config import load_data_config
from asag.data.splits import _bin_scores
from asag.models.data import load_bundle, make_X, make_y
from asag.models.fusion import LgbmFusionHead
from asag.models.metrics import compute_metrics
from asag.models.tasks import get_spec

KFOLD_DATASETS = ["mohler", "powergrading", "mindreading"]
SEEDS = [42, 1, 2]
# One fixed regularized head for every cell (fair stratified-vs-grouped compare).
HEAD = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.0, reg_lambda=0.0)


def grouped_folds(df: pd.DataFrame, k: int, seed: int) -> np.ndarray:
    """Hold out whole question_id groups; stratify on score-bins/labels."""
    spec = get_spec(df["dataset"].iloc[0]) if "dataset" in df else None
    if spec is not None and spec.task_type == "classification":
        y = df["label"].astype("category").cat.codes.to_numpy()
    else:
        y = _bin_scores(df["score"]).to_numpy()
    groups = df["question_id"].astype(str).to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=int)
    for fi, (_, te) in enumerate(sgkf.split(np.zeros(len(df)), y, groups)):
        fold[te] = fi
    return fold


def qmean_predict(tr: pd.DataFrame, te: pd.DataFrame, task: str,
                  vocab: dict) -> np.ndarray:
    """Per-question shortcut: train question-mean (reg/ord) or majority (clf)."""
    if task == "classification":
        codes = tr["label"].astype(str).map(vocab)
        glob = int(codes.dropna().mode().iloc[0]) if codes.notna().any() else 0
        per_q = (tr.assign(_c=codes).dropna(subset=["_c"])
                 .groupby(tr["question_id"].astype(str))["_c"]
                 .agg(lambda s: s.mode().iloc[0]).astype(int).to_dict())
    else:
        sc = pd.to_numeric(tr["score"], errors="coerce")
        glob = float(sc.mean())
        per_q = sc.groupby(tr["question_id"].astype(str)).mean().to_dict()
    return np.array([per_q.get(q, glob) for q in te["question_id"].astype(str)],
                    dtype=float)


def run_strategy(bundle, fold_of, headline) -> dict:
    """Pooled-OOF GBM + qmean shortcut over the given fold assignment."""
    df, spec, vocab = bundle.df, bundle.spec, bundle.label_vocab
    y_all = make_y(df, bundle)
    finite = np.isfinite(y_all)
    gbm_scores, sc_scores = [], []
    for seed in SEEDS:
        fold = fold_of(seed)
        yt, yp_g, yp_s = [], [], []
        for f in sorted(set(fold[fold >= 0])):
            te_m = (fold == f) & finite
            tr_m = (fold != f) & (fold >= 0) & finite
            tr, te = df[tr_m], df[te_m]
            if tr.empty or te.empty:
                continue
            Xtr, ytr = make_X(tr, bundle.feature_cols), make_y(tr, bundle)
            Xte, yte = make_X(te, bundle.feature_cols), make_y(te, bundle)
            head = LgbmFusionHead(spec.task_type, _CFG, seed).fit(Xtr, ytr)
            yt.append(yte); yp_g.append(head.predict(Xte))
            yp_s.append(qmean_predict(tr, te, spec.task_type, vocab))
        YT = np.concatenate(yt)
        gbm_scores.append(compute_metrics(YT, np.concatenate(yp_g), spec.metrics)[headline])
        sc_scores.append(compute_metrics(YT, np.concatenate(yp_s), spec.metrics)[headline])
    return {
        "gbm_mean": round(float(np.nanmean(gbm_scores)), 4),
        "gbm_std": round(float(np.nanstd(gbm_scores)), 4),
        "qshortcut_mean": round(float(np.nanmean(sc_scores)), 4),
        "n_questions": int(df["question_id"].astype(str).nunique()),
    }


def main() -> None:
    global _CFG
    from asag.config import LightGBMCfg
    _CFG = LightGBMCfg(**HEAD)
    cfg = load_data_config()
    out = {"head": HEAD, "seeds": SEEDS, "datasets": {}}
    for name in KFOLD_DATASETS:
        spec = get_spec(name)
        bundle = load_bundle(name, cfg, spec)
        if bundle is None:
            print(f"{name}: no features.parquet, skip"); continue
        headline = spec.headline
        legacy = bundle.df["fold"].to_numpy()
        res = {
            "headline_metric": headline,
            "stratified_legacy": run_strategy(bundle, lambda s: legacy, headline),
            "grouped_by_question": run_strategy(
                bundle, lambda s: grouped_folds(bundle.df, cfg.splits.cv_k_folds, s), headline),
        }
        sg = res["grouped_by_question"]["gbm_mean"]
        st = res["stratified_legacy"]["gbm_mean"]
        res["gbm_inflation"] = round(st - sg, 4)
        out["datasets"][name] = res
        print(f"\n=== {name} ({headline}, {res['stratified_legacy']['n_questions']} questions) ===")
        print(f"  GBM   stratified-legacy : {st:.4f}")
        print(f"  GBM   grouped-by-question: {sg:.4f}   (inflation = {res['gbm_inflation']:+.4f})")
        print(f"  qshortcut stratified     : {res['stratified_legacy']['qshortcut_mean']:.4f}")
        print(f"  qshortcut grouped        : {res['grouped_by_question']['qshortcut_mean']:.4f}")

    dst = cfg.paths.reports / "phase4_audit"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "grouped_cv_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dst / 'grouped_cv_audit.json'}")


if __name__ == "__main__":
    main()
