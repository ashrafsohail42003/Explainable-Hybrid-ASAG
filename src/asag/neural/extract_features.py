"""Phase 4 — out-of-fold DeBERTa signals for the hybrid (run on Colab/GPU).

Produces, per dataset, a ``neural_oof.parquet`` that is **row-aligned to
``features.parquet``** and holds strictly out-of-fold cross-encoder signals — no
row is ever scored by a model that trained on it, so feeding these into the
LightGBM head (``data.load_bundle`` concatenates any ``neural_*`` columns) does
not leak. This is the *neural-as-feature* hybrid, not late fusion: the GBM keeps
its TreeSHAP / concept-attribution explainability and gains the semantic signal
the handcrafted features lack.

OOF construction mirrors the GBM eval protocol exactly:

* **kfold** datasets — grouped ``fold`` column: each row is predicted by the model
  trained on the other folds (one pass = full coverage).
* **official_split** datasets — test/dev rows are predicted by a model trained on
  ``train``; the ``train`` rows themselves get an **inner grouped k-fold** OOF pass
  (so the GBM never sees in-sample neural predictions for its own training rows).

Emitted columns: ``neural_score`` (regression raw / ordinal expected-rank /
classification max-prob — the ``y_cont`` from :func:`asag.neural.trainer.fit_predict`)
and ``neural_pred`` (the discrete metric-space prediction).

Colab usage: see ``notebooks/02_neural_colab.ipynb`` — it installs torch/transformers
(+peft for LoRA), runs this module, and downloads ``neural_oof.parquet`` per dataset
back into ``data/processed/<name>/``.  ``python -m asag.neural.extract_features [names]``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from asag.config import DataConfig, load_data_config
from asag.data.splits import make_grouped_kfold
from asag.models.tasks import REGISTRY, TaskSpec, get_spec
from asag.neural.evaluate_neural import load_text_df
from asag.neural.trainer import fit_predict
from asag.utils.logging import get_logger

log = get_logger()


def _max_len(name: str, cfg: DataConfig) -> int:
    return cfg.neural.max_len_overrides.get(name, cfg.neural.max_len)


def _oof_kfold(name: str, df: pd.DataFrame, spec: TaskSpec, cfg: DataConfig, tok,
               seed: int, max_len: int) -> pd.DataFrame:
    score = np.full(len(df), np.nan)
    pred = np.full(len(df), np.nan)
    folds = sorted(int(f) for f in df["fold"].unique() if int(f) >= 0)
    for f in folds:
        tr = df[(df["fold"] != f) & (df["fold"] >= 0)].reset_index(drop=True)
        te_idx = np.where(df["fold"].to_numpy() == f)[0]
        te = df.iloc[te_idx].reset_index(drop=True)
        if tr.empty or te.empty:
            continue
        r = fit_predict(tr, te, spec, cfg.neural, seed, max_len=max_len, tokenizer=tok)
        score[te_idx] = r["y_cont"]
        pred[te_idx] = r["y_pred"]
    return pd.DataFrame({"neural_score": score, "neural_pred": pred}, index=df.index)


def _oof_official(name: str, df: pd.DataFrame, spec: TaskSpec, cfg: DataConfig, tok,
                  seed: int, max_len: int) -> pd.DataFrame:
    score = np.full(len(df), np.nan)
    pred = np.full(len(df), np.nan)
    is_train = (df["split"] == "train").to_numpy()
    train_df = df[is_train].reset_index(drop=True)
    dev_df = df[df["split"] == "dev"].reset_index(drop=True)

    # test/dev rows: one model trained on the full train split
    for split in [s for s in df["split"].unique() if s not in ("train",)]:
        te_idx = np.where(df["split"].to_numpy() == split)[0]
        te = df.iloc[te_idx].reset_index(drop=True)
        if te.empty or train_df.empty:
            continue
        r = fit_predict(train_df, te, spec, cfg.neural, seed, max_len=max_len,
                        dev_df=dev_df if len(dev_df) else None, tokenizer=tok)
        score[te_idx] = r["y_cont"]; pred[te_idx] = r["y_pred"]

    # train rows: inner grouped k-fold OOF so the GBM gets honest train-row signals
    if not train_df.empty:
        inner = make_grouped_kfold(train_df, k=cfg.splits.cv_k_folds, seed=seed,
                                   group_col="question_id").to_numpy()
        train_pos = np.where(is_train)[0]
        for f in sorted(set(inner)):
            tr = train_df[inner != f].reset_index(drop=True)
            te_loc = np.where(inner == f)[0]
            te = train_df.iloc[te_loc].reset_index(drop=True)
            if tr.empty or te.empty:
                continue
            r = fit_predict(tr, te, spec, cfg.neural, seed, max_len=max_len, tokenizer=tok)
            score[train_pos[te_loc]] = r["y_cont"]
            pred[train_pos[te_loc]] = r["y_pred"]
    return pd.DataFrame({"neural_score": score, "neural_pred": pred}, index=df.index)


def extract_dataset(name: str, cfg: DataConfig, tok=None) -> pd.DataFrame | None:
    spec = get_spec(name)
    df = load_text_df(name, cfg)
    if df is None:
        log.warning(f"{name}: encoder.parquet missing — run `make preprocess`; skipping")
        return None
    # Fail loudly rather than emit an all-NaN OOF that load_bundle would silently ingest.
    if spec.protocol == "kfold":
        if "fold" not in df.columns or not (df["fold"].to_numpy() >= 0).any():
            log.error(f"{name}: kfold dataset but no valid 'fold' column in encoder.parquet "
                      f"— re-run `make preprocess`; skipping")
            return None
    elif not (df["split"].to_numpy() == "train").any():
        log.error(f"{name}: official-split dataset has no 'train' rows — cannot build OOF; skipping")
        return None
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.neural.backbone)
    max_len = _max_len(name, cfg)
    seed = cfg.neural.seeds[0]
    fn = _oof_kfold if spec.protocol == "kfold" else _oof_official
    oof = fn(name, df, spec, cfg, tok, seed, max_len)
    oof.insert(0, "question_id", df["question_id"].values)   # alignment guard for load_bundle

    out_path = cfg.paths.processed / name / "neural_oof.parquet"
    oof.to_parquet(out_path, index=False)
    cov = float(np.isfinite(oof["neural_score"]).mean())
    (log.warning if cov < 0.99 else log.info)(
        f"{name}: wrote {out_path} (OOF coverage {cov:.2%}"
        + ("" if cov >= 0.99 else " — some rows uncovered; check fold/split assignment)"))
    return oof


def run_all(cfg: DataConfig | None = None, names: list[str] | None = None) -> None:
    cfg = cfg or load_data_config()
    names = names or [n for n in REGISTRY
                      if (cfg.paths.processed / n / "encoder.parquet").exists()]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.neural.backbone)
    for name in names:
        extract_dataset(name, cfg, tok)


if __name__ == "__main__":
    import sys
    run_all(names=sys.argv[1:] or None)
