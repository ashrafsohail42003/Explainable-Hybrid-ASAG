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
from asag.data.loaders import (
    UNIFIED_COLUMNS,
    load_mindreading,
    load_mohler,
    load_powergrading,
    load_saf,
    load_semeval,
)


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


def test_powergrading_loader(cfg):
    pg = cfg.datasets.get("powergrading")
    if pg is None or not pg.enabled:
        pytest.skip("powergrading disabled in config.")
    pg_dir = cfg.paths.raw / pg.raw_subdir
    if not pg_dir.exists() or not (pg_dir / "studentanswers_grades_698.tsv").exists():
        pytest.skip("Powergrading raw not present — skipping.")
    df = load_powergrading(cfg)
    _assert_schema(df, "powergrading")
    assert (df["domain"] == "civics").all()
    assert df["question_id"].nunique() == 20
    s = df["score"].dropna()
    assert s.between(0.0, 1.0).all(), f"PG score out of [0,1]: [{s.min()}, {s.max()}]"
    assert set(df["label"].unique()).issubset({"correct", "incorrect"})


def test_mindreading_loader(cfg):
    mr = cfg.datasets.get("mindreading")
    if mr is None or not mr.enabled:
        pytest.skip("mindreading disabled in config.")
    mr_dir = cfg.paths.raw / mr.raw_subdir
    if not mr_dir.exists() or not any(mr_dir.glob("*.xlsx")):
        pytest.skip("MindReading raw not present — skipping.")
    df = load_mindreading(cfg)
    _assert_schema(df, "mindreading")
    assert (df["domain"] == "mindreading_behavioral").all()
    assert df["question_id"].nunique() == 11, f"expected 11 tasks, got {df['question_id'].nunique()}"
    s = df["score"].dropna()
    assert set(s.unique()).issubset({0.0, 1.0, 2.0}), f"unexpected score values: {set(s.unique())}"
    # reference_answer is intentionally empty for this dataset
    assert (df["reference_answer"].astype(str) == "").all()


def test_mohler_loader(cfg):
    moh_dir = cfg.paths.raw / cfg.datasets["mohler"].raw_subdir
    canonical = moh_dir / "mohler_canonical_from_asag2024.parquet"
    if not canonical.exists():
        pytest.skip("Mohler canonical parquet not present — skipping.")
    df = load_mohler(cfg)
    _assert_schema(df, "mohler")
    assert (df["domain"] == "cs_data_structures").all()
    s = df["score"].dropna()
    # ASAG2024 uses 0..100 (normalized x 100). Allow 0..100 to cover both raw and 0..5 cases.
    assert s.between(0, 100).all(), f"Mohler score out of [0,100]: [{s.min()}, {s.max()}]"
