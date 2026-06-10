"""Phase 4 — Leave-One-Dataset-Out (LODO) cross-domain transfer.

The six datasets are heterogeneous (5-way / binary / ordinal / regression), so a
single transfer task is only well-defined on a **unified binary** target
``is_correct``. We map every dataset onto that target (see :func:`binarize_target`),
train on the union of five datasets' **shared** interpretable features, and score
the held-out sixth — a genuine cross-domain generalization test (no question and
no domain seen in training).

This is the GBM-on-shared-features LODO (runnable, NaN-native — datasets without a
reference answer simply carry NaN in the 24 reference-dependent columns). The
neural cross-encoder LODO (train a DeBERTa grader on five corpora, zero-shot the
sixth) is the natural text-transfer counterpart and runs on Colab; its predictions
drop into the same ``score_lodo`` harness via a ``neural_oof``-style cache.

``python -m asag.models.lodo`` → ``reports/phase4_lodo/lodo.json`` (+ figure).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from asag.config import DataConfig, load_data_config
from asag.models.data import Bundle, load_bundle, make_X
from asag.models.fusion import LIGHTGBM_AVAILABLE, LgbmFusionHead
from asag.models.tasks import REGISTRY, get_spec
from asag.utils.logging import get_logger
from asag.utils.seed import set_global_seed

log = get_logger()

# How each dataset's native target collapses to a binary "is the answer correct?".
# label-based datasets: only the fully-correct class is positive.
LABEL_POSITIVE = {"semeval": {"correct"}, "powergrading": {"correct"}}


def binarize_target(df: pd.DataFrame, name: str) -> np.ndarray:
    """Unified binary ``is_correct`` (1 = correct). NaN rows are dropped upstream.

    * label datasets (semeval, powergrading): 1 iff label in the positive set.
    * score datasets (saf, mohler, asap_sas, mindreading): 1 iff
      ``score >= midpoint(scale)`` where the scale is read from the dataset itself.
    """
    if name in LABEL_POSITIVE:
        pos = LABEL_POSITIVE[name]
        return df["label"].astype(str).isin(pos).to_numpy(dtype=float)
    s = pd.to_numeric(df["score"], errors="coerce")
    lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
    mid = (lo + hi) / 2.0
    out = (s >= mid).astype(float)
    out[s.isna().to_numpy()] = np.nan
    return out.to_numpy(dtype=float)


def _frame(name: str, cfg: DataConfig, shared: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    bundle = load_bundle(name, cfg, get_spec(name))
    df = bundle.df
    y = binarize_target(df, name)
    m = np.isfinite(y)
    return df.loc[m, shared].astype("float64"), y[m]


def _shared_features(cfg: DataConfig, names: list[str]) -> list[str]:
    cols: set | None = None
    for n in names:
        b = load_bundle(n, cfg, get_spec(n))
        fc = set(c for c in b.feature_cols if not c.startswith("neural_"))
        cols = fc if cols is None else (cols & fc)
    return sorted(cols or [])


def score_lodo(cfg: DataConfig, names: list[str], seed: int) -> dict:
    shared = _shared_features(cfg, names)
    frames = {n: _frame(n, cfg, shared) for n in names}
    out: dict[str, dict] = {}
    for held in names:
        Xtr = pd.concat([frames[n][0] for n in names if n != held], ignore_index=True)
        ytr = np.concatenate([frames[n][1] for n in names if n != held])
        Xte, yte = frames[held]
        if np.unique(ytr).size < 2 or np.unique(yte).size < 2:
            out[held] = {"status": "degenerate", "n_test": int(len(yte))}
            continue
        set_global_seed(seed)
        head = LgbmFusionHead("classification", cfg.model.lightgbm, seed).fit(Xtr, ytr)
        proba = head.model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        out[held] = {
            "status": "ok",
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "pos_rate_test": round(float(np.mean(yte)), 4),
            "macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
            "auc": round(float(roc_auc_score(yte, proba)), 4),
        }
    return {"shared_features": shared, "per_dataset": out}


def run_all(cfg: DataConfig | None = None, names: list[str] | None = None) -> dict:
    cfg = cfg or load_data_config()
    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError("lightgbm is not installed; run `uv pip install lightgbm`")
    names = names or [n for n in REGISTRY
                      if (cfg.paths.processed / n / "features.parquet").exists()]
    per_seed = [score_lodo(cfg, names, s) for s in cfg.model.seeds]

    # aggregate macro_f1 / auc mean±std across seeds
    agg: dict[str, dict] = {}
    for held in names:
        cells = [ps["per_dataset"][held] for ps in per_seed
                 if ps["per_dataset"][held].get("status") == "ok"]
        if not cells:
            agg[held] = per_seed[0]["per_dataset"][held]
            continue
        agg[held] = {
            "status": "ok",
            "n_train": cells[0]["n_train"], "n_test": cells[0]["n_test"],
            "pos_rate_test": cells[0]["pos_rate_test"],
            "macro_f1_mean": round(float(np.mean([c["macro_f1"] for c in cells])), 4),
            "macro_f1_std": round(float(np.std([c["macro_f1"] for c in cells])), 4),
            "auc_mean": round(float(np.mean([c["auc"] for c in cells])), 4),
            "auc_std": round(float(np.std([c["auc"] for c in cells])), 4),
        }
    doc = {"target": "unified_binary_is_correct", "model": "gbm_shared_features",
           "seeds": list(cfg.model.seeds),
           "shared_features": per_seed[0]["shared_features"],
           "label_positive": {k: sorted(v) for k, v in LABEL_POSITIVE.items()},
           "results": agg}

    dst = cfg.paths.reports / "phase4_lodo"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "lodo.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    log.info(f"wrote {dst / 'lodo.json'}")
    for held, r in agg.items():
        if r.get("status") == "ok":
            log.info(f"  LODO hold-out {held}: macro_f1={r['macro_f1_mean']:.4f} "
                     f"auc={r['auc_mean']:.4f} (pos={r['pos_rate_test']})")
    return doc


if __name__ == "__main__":
    import sys
    run_all(names=sys.argv[1:] or None)
