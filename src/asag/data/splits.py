"""Official splits + k-fold scaffolding.

  * ``get_official_splits(df)``: dict of {split_name -> DataFrame} restricted
    to the official splits already present in ``df['split']``.
  * ``make_grouped_kfold(df, k, seed, group_col='question_id')``: the **primary**
    CV for datasets without an official test split — holds out whole
    ``question_id`` groups so the same question never appears in both train and
    test (a *leave-questions-out* / unseen-question protocol). Stratifies the
    held-out groups on score bins (or labels) for balance.
  * ``make_stratified_kfold(df, k, seed, stratify_on='score')``: the legacy
    score-stratified k-fold. **Leaks ``question_id`` across folds** (a question's
    rows are split between train and test) so it over-estimates generalization;
    kept only for the inflation audit and as the seen-question *upper bound*.

Both return a Series of fold indices aligned with ``df.index``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold


def get_official_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return one DataFrame per non-empty split label in ``df['split']``."""
    out: dict[str, pd.DataFrame] = {}
    for sp in sorted(df["split"].unique()):
        if not sp:
            continue
        out[sp] = df.loc[df["split"] == sp].reset_index(drop=True)
    return out


def _bin_scores(scores: pd.Series, n_bins: int = 6) -> pd.Series:
    """Bin continuous/ordinal scores into integer classes for stratification."""
    s = pd.to_numeric(scores, errors="coerce")
    if s.notna().sum() == 0:
        # no usable score → all in bin 0; CV becomes a plain k-fold
        return pd.Series(np.zeros(len(s), dtype=int), index=s.index)
    s_min, s_max = float(s.min()), float(s.max())
    if s_min == s_max:
        return pd.Series(np.zeros(len(s), dtype=int), index=s.index)
    edges = np.linspace(s_min, s_max, n_bins + 1)
    binned = pd.cut(s, bins=edges, labels=False, include_lowest=True).fillna(0).astype(int)
    return binned


def make_stratified_kfold(
    df: pd.DataFrame,
    k: int = 5,
    seed: int = 42,
    stratify_on: str = "score",
) -> pd.Series:
    """Return a Series of fold indices in [0, k) aligned with ``df.index``."""
    if k < 2:
        raise ValueError("k must be >= 2")
    n = len(df)
    if n < k:
        raise ValueError(f"cannot create {k} folds for {n} rows")

    if stratify_on == "score":
        y = _bin_scores(df["score"])
    else:
        y = df[stratify_on].astype("category").cat.codes

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = np.full(n, -1, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(skf.split(np.zeros(n), y.values)):
        folds[test_idx] = fold_idx
    return pd.Series(folds, index=df.index, name="fold")


def make_grouped_kfold(
    df: pd.DataFrame,
    k: int = 5,
    seed: int = 42,
    group_col: str = "question_id",
    stratify_on: str = "score",
) -> pd.Series:
    """Leave-questions-out k-fold: no ``group_col`` value spans train and test.

    Holds out whole question groups (the unseen-question protocol). The folds are
    score-/label-stratified where possible via ``StratifiedGroupKFold``; if there
    are fewer groups than ``k`` we fall back to ``GroupKFold(min(k, n_groups))``
    (purely grouped, no stratification). Returns fold indices aligned to ``df.index``.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    n = len(df)
    if n < k:
        raise ValueError(f"cannot create {k} folds for {n} rows")

    groups = df[group_col].astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    if stratify_on == "score":
        y = _bin_scores(df["score"]).to_numpy()
    else:
        y = df[stratify_on].astype("category").cat.codes.to_numpy()

    folds = np.full(n, -1, dtype=int)
    if n_groups < k:
        k_eff = max(2, min(k, n_groups))
        import logging
        logging.getLogger("asag").warning(
            f"make_grouped_kfold: only {n_groups} groups for k={k}; "
            f"using GroupKFold(n_splits={k_eff}) without stratification"
        )
        gkf = GroupKFold(n_splits=k_eff)
        splitter = gkf.split(np.zeros(n), groups=groups)
    else:
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        splitter = sgkf.split(np.zeros(n), y, groups)

    for fold_idx, (_, test_idx) in enumerate(splitter):
        folds[test_idx] = fold_idx
    return pd.Series(folds, index=df.index, name="fold")
