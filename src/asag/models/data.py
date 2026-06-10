"""Assemble model inputs from the Phase 2B ``features.parquet``.

The feature matrix is every column that is not a key column; NaN values are left
intact (the GBM head is NaN-native, so the 25 reference-dependent features that
are NaN for ASAP-SAS / MIND-CA need no imputation). Classification targets are
label-encoded against a dataset-wide vocabulary so train/test codes agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from asag.config import DataConfig
from asag.models.tasks import TaskSpec

# Mirrors asag.features.build.KEY_COLUMNS (the non-feature columns).
KEY_COLUMNS = ["question_id", "score", "label", "dataset", "domain", "split", "fold"]


@dataclass
class Bundle:
    name: str
    df: pd.DataFrame
    feature_cols: list[str]
    spec: TaskSpec
    label_vocab: dict[str, int]   # {} for non-classification tasks


def load_bundle(name: str, cfg: DataConfig, spec: TaskSpec) -> Bundle | None:
    """Read ``features.parquet`` for ``name``; return None if it is absent.

    If ``neural_oof.parquet`` (out-of-fold DeBERTa cross-encoder signals produced
    on Colab/GPU) sits beside it, its columns are concatenated as extra features —
    this is the hybrid. The file is positionally row-aligned to ``features.parquet``
    (same builder order); we assert equal length + identical ``question_id`` order
    rather than ``merge`` (question_id is not unique), mirroring ``build._merge_views``.
    """
    path = cfg.paths.processed / name / "features.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path).reset_index(drop=True)

    neural_path = cfg.paths.processed / name / "neural_oof.parquet"
    if neural_path.exists():
        ndf = pd.read_parquet(neural_path).reset_index(drop=True)
        if len(ndf) != len(df):
            raise ValueError(f"{name}: neural_oof rows {len(ndf)} != features rows {len(df)}")
        if "question_id" in ndf.columns and not df["question_id"].astype(str).equals(
                ndf["question_id"].astype(str)):
            raise ValueError(f"{name}: neural_oof question_id order differs from features")
        ncols = [c for c in ndf.columns if c.startswith("neural_")]
        df = pd.concat([df, ndf[ncols].reset_index(drop=True)], axis=1)

    feature_cols = [c for c in df.columns if c not in KEY_COLUMNS]
    vocab: dict[str, int] = {}
    if spec.task_type == "classification":
        labels = sorted(s for s in df["label"].astype(str).unique() if s != "")
        vocab = {lab: i for i, lab in enumerate(labels)}
    return Bundle(name=name, df=df, feature_cols=feature_cols, spec=spec, label_vocab=vocab)


QPRIOR_COL = "qprior_train_mean"


def question_prior(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   spec: TaskSpec) -> tuple[np.ndarray, np.ndarray]:
    """Fold-safe per-question difficulty prior (leakage-aware target encoding).

    Returns ``(train_prior, test_prior)``:

    * train rows get a **leave-one-out** per-question mean (a row never sees its
      own target), singletons fall back to the global train mean;
    * test rows get the plain per-question train mean, **NaN for unseen
      questions** — so under grouped CV the prior is absent (NaN-native head
      ignores it) and cannot leak question identity.

    Only meaningful for ordinal/regression (a numeric difficulty signal); returns
    all-NaN for classification (grouped CV already removes its question leakage).
    """
    n_tr, n_te = len(train_df), len(test_df)
    if spec.task_type == "classification":
        return np.full(n_tr, np.nan), np.full(n_te, np.nan)
    qid_tr = train_df["question_id"].astype(str)
    y_tr = pd.to_numeric(train_df["score"], errors="coerce")
    grp = y_tr.groupby(qid_tr)
    gsum, gcnt = grp.transform("sum"), grp.transform("count")
    loo = (gsum - y_tr) / (gcnt - 1)
    global_mean = float(y_tr.mean()) if y_tr.notna().any() else np.nan
    train_prior = loo.fillna(global_mean).to_numpy(dtype=float)
    per_q = grp.mean().to_dict()
    test_prior = test_df["question_id"].astype(str).map(per_q).to_numpy(dtype=float)
    return train_prior, test_prior


def make_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Feature matrix as float; NaN preserved for the NaN-native head."""
    return df[feature_cols].astype("float64")


def make_y(df: pd.DataFrame, bundle: Bundle) -> np.ndarray:
    """Target vector: encoded class codes (classification) or float score."""
    spec = bundle.spec
    if spec.task_type == "classification":
        codes = df["label"].astype(str).map(bundle.label_vocab)
        return codes.to_numpy(dtype=float)   # may contain NaN for unseen/empty labels
    return pd.to_numeric(df["score"], errors="coerce").to_numpy(dtype=float)


def valid_rows(df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Boolean mask of rows whose target is finite (drops unlabelled rows)."""
    return np.isfinite(y)
