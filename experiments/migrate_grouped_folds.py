"""One-off migration: rewrite the ``fold`` column from legacy score-stratified to
grouped-by-question, in place, for the no-official-split datasets — WITHOUT
recomputing features (SBERT untouched; only the fold assignment changes).

For each of mohler/powergrading/mindreading it recomputes grouped folds once from
``features.parquet`` (the canonical row order), asserts the other two views
(``encoder.parquet``, ``feature.parquet``) are row-aligned by ``question_id``, and
writes the same fold array into all three parquets (+ jsonl backups). Idempotent.

Run (repo root):
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
    "C:/Users/MSI/.cache/asag-venvs/asag-py311/Scripts/python.exe" \
    experiments/migrate_grouped_folds.py
"""
from __future__ import annotations

import pandas as pd

from asag.config import load_data_config
from asag.data.splits import make_grouped_kfold

KFOLD_DATASETS = ["mohler", "powergrading", "mindreading"]
VIEWS = ["features.parquet", "feature.parquet", "encoder.parquet"]


def main() -> None:
    cfg = load_data_config()
    for name in KFOLD_DATASETS:
        ddir = cfg.paths.processed / name
        canon = pd.read_parquet(ddir / "features.parquet").reset_index(drop=True)
        new_fold = make_grouped_kfold(
            canon, k=cfg.splits.cv_k_folds, seed=cfg.seed,
            group_col="question_id", stratify_on=cfg.splits.stratify_on).to_numpy()
        n_groups = canon["question_id"].astype(str).nunique()
        print(f"{name}: {len(canon)} rows, {n_groups} questions, "
              f"folds={sorted(set(new_fold))}")

        for view in VIEWS:
            p = ddir / view
            if not p.exists():
                print(f"  skip missing {view}"); continue
            df = pd.read_parquet(p).reset_index(drop=True)
            assert len(df) == len(canon), f"{name}/{view}: length mismatch"
            assert df["question_id"].astype(str).equals(
                canon["question_id"].astype(str)), f"{name}/{view}: qid order differs"
            df["fold"] = new_fold
            df.to_parquet(p, index=False)
            jsonl = p.with_suffix(".jsonl")
            if jsonl.exists():
                df.to_json(jsonl, orient="records", lines=True, force_ascii=False)
            print(f"  rewrote fold in {view}")
    print("migration done")


if __name__ == "__main__":
    main()
