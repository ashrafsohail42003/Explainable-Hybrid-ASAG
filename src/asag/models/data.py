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
    """Read ``features.parquet`` for ``name``; return None if it is absent."""
    path = cfg.paths.processed / name / "features.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in KEY_COLUMNS]
    vocab: dict[str, int] = {}
    if spec.task_type == "classification":
        labels = sorted(s for s in df["label"].astype(str).unique() if s != "")
        vocab = {lab: i for i, lab in enumerate(labels)}
    return Bundle(name=name, df=df, feature_cols=feature_cols, spec=spec, label_vocab=vocab)


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
