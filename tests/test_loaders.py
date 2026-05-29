"""Smoke tests for the unified loaders.

These tests check structural invariants without needing the actual data to be
present: when a dataset's raw files are missing, the test is skipped (so it
runs on a clean clone). When the data IS present, it verifies the schema,
non-empty rows, score sanity, and presence of the official splits.
"""

from __future__ import annotations

import pandas as pd
import pytest

from asag.config import load_data_config
from asag.data.loaders import UNIFIED_COLUMNS, load_mohler, load_saf, load_semeval


@pytest.fixture(scope="module")
def cfg():
    return load_data_config()


def _assert_schema(df: pd.DataFrame, expected_dataset_prefix: str) -> None:
    assert list(df.columns) == UNIFIED_COLUMNS, df.columns.tolist()
    assert len(df) > 0, "loader returned no rows"
    assert df["question"].str.len().gt(0).any(), "all questions empty"
    assert df["student_answer"].str.len().gt(0).any(), "all student answers empty"
    assert df["dataset"].str.startswith(expected_dataset_prefix).any()


def test_semeval_loader(cfg):
    raw_dir = cfg.paths.raw / cfg.datasets["semeval"].raw_subdir
    if not raw_dir.exists():
        pytest.skip("SemEval raw not present — skipping (run `make download`).")
    df = load_semeval(cfg)
    _assert_schema(df, "semeval_")
    # at least one of the official splits must be present
    expected_splits = {"train", "test_ua", "test_uq"}
    found = set(df["split"].unique())
    assert expected_splits.issubset(found), f"semeval missing splits: {expected_splits - found}"


def test_saf_loader(cfg):
    saf_dir = cfg.paths.raw / cfg.datasets["saf"].raw_subdir
    if not saf_dir.exists() or not any(saf_dir.glob("*.parquet")):
        pytest.skip("SAF raw not present — skipping.")
    df = load_saf(cfg)
    _assert_schema(df, "saf_comm_nets")
    assert (df["domain"] == "comm_networks").all()
    # official SAF splits map to train/dev/test_ua/test_uq
    assert {"train", "dev", "test_ua", "test_uq"}.issubset(df["split"].unique())
    # score should be a valid float in [0, 4]
    s = df["score"].dropna()
    assert s.between(0, 4).all(), f"SAF score range out of bounds: [{s.min()}, {s.max()}]"


def test_mohler_loader(cfg):
    moh_dir = cfg.paths.raw / cfg.datasets["mohler"].raw_subdir
    if not moh_dir.exists() or not any(moh_dir.rglob("*.csv")):
        pytest.skip("Mohler raw not present — skipping.")
    df = load_mohler(cfg)
    _assert_schema(df, "mohler")
    assert (df["domain"] == "cs_data_structures").all()
    s = df["score"].dropna()
    assert s.between(0, 5).all(), f"Mohler score out of [0,5]: [{s.min()}, {s.max()}]"
