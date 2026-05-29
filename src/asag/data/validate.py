"""Schema, leakage, duplicate, and missing-value validation.

Usage::

    python -m asag.data.validate

Emits one JSON report per dataset under ``reports/validation/`` plus a
combined ``reports/validation/summary.json``.

Checks per dataset:
  * ``check_schema``: required columns + dtypes.
  * ``check_missing``: counts NaN/empty in required text columns + score.
  * ``check_duplicates``: exact duplicates on (question_id, student_answer)
    and per-question near-duplicates on the student answer using a
    token-set Jaccard threshold (default 0.95).
  * ``check_leakage``: same student_answer appearing across train and test
    splits, and same question_id appearing across train and test_uq/UD
    (which is structurally illegal for those splits).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from asag.config import DataConfig, ensure_dirs, load_data_config
from asag.data.loaders import UNIFIED_COLUMNS, load_all
from asag.utils.logging import get_logger
from asag.utils.seed import set_global_seed

log = get_logger()


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9']+", _norm_text(s)) if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def check_schema(df: pd.DataFrame) -> dict:
    missing = [c for c in UNIFIED_COLUMNS if c not in df.columns]
    extras = [c for c in df.columns if c not in UNIFIED_COLUMNS]
    ok = not missing and not extras
    return {"ok": ok, "missing_columns": missing, "extra_columns": extras,
            "n_rows": int(len(df)), "n_cols": int(df.shape[1])}


def check_missing(df: pd.DataFrame) -> dict:
    out: dict[str, Any] = {}
    for col in ("question", "reference_answer", "student_answer"):
        empty = (df[col].astype(str).str.strip() == "").sum()
        out[col + "_empty"] = int(empty)
    out["score_nan"] = int(df["score"].isna().sum())
    out["label_empty"] = int((df["label"].astype(str).str.strip() == "").sum())
    return out


def check_duplicates(df: pd.DataFrame, near_threshold: float = 0.95) -> dict:
    """Exact dup on (question_id, student_answer) + per-question near-dup pairs."""
    dup_mask = df.duplicated(subset=["question_id", "student_answer"], keep="first")
    exact = int(dup_mask.sum())

    near_pairs = 0
    samples: list[dict] = []
    grouped = df.groupby("question_id", sort=False)
    for qid, sub in grouped:
        if len(sub) < 2:
            continue
        toks = [_tokens(s) for s in sub["student_answer"].tolist()]
        idxs = sub.index.tolist()
        n = len(toks)
        for i in range(n):
            for j in range(i + 1, n):
                if not toks[i] or not toks[j]:
                    continue
                j_sim = _jaccard(toks[i], toks[j])
                if j_sim >= near_threshold and j_sim < 1.0:
                    near_pairs += 1
                    if len(samples) < 5:
                        samples.append({
                            "question_id": str(qid),
                            "jaccard": round(j_sim, 4),
                            "a_index": int(idxs[i]),
                            "b_index": int(idxs[j]),
                        })
    return {
        "exact_duplicate_rows": exact,
        "near_duplicate_pairs_threshold": near_threshold,
        "near_duplicate_pairs": near_pairs,
        "near_duplicate_samples": samples,
    }


def check_leakage(df: pd.DataFrame) -> dict:
    """Detect student_answer leakage train<->test_* and question_id leakage train<->test_uq/test_ud."""
    splits = set(df["split"].unique())
    train_mask = df["split"] == "train"
    if not train_mask.any() or "all" in splits and splits == {"all"}:
        return {"applicable": False, "reason": "no official train split"}

    train_ans = set(_norm_text(s) for s in df.loc[train_mask, "student_answer"])
    train_qids = set(df.loc[train_mask, "question_id"])

    leaks_ans: dict[str, int] = {}
    leaks_qid: dict[str, int] = {}
    for sp in sorted(splits - {"train"}):
        sub = df[df["split"] == sp]
        ans_overlap = sum(1 for s in sub["student_answer"] if _norm_text(s) in train_ans)
        leaks_ans[sp] = int(ans_overlap)
        if sp in {"test_uq", "test_ud"}:
            qid_overlap = sub["question_id"].isin(train_qids).sum()
            leaks_qid[sp] = int(qid_overlap)
    return {"applicable": True,
            "student_answer_overlap_with_train": leaks_ans,
            "question_id_overlap_with_train_in_unseen_splits": leaks_qid}


def validate_dataset(df: pd.DataFrame, name: str, near_threshold: float) -> dict:
    log.info(f"validating {name} ({len(df)} rows)")
    return {
        "name": name,
        "schema": check_schema(df),
        "missing": check_missing(df),
        "duplicates": check_duplicates(df, near_threshold=near_threshold),
        "leakage": check_leakage(df),
    }


def _write_report(out_dir: Path, name: str, report: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return p


def run_all(cfg: DataConfig) -> dict:
    ensure_dirs(cfg)
    set_global_seed(cfg.seed)
    out_dir = cfg.paths.reports / "validation"
    near = cfg.validation.near_duplicate_jaccard_threshold

    frames = load_all(cfg)
    all_reports: dict[str, dict] = {}
    for name, df in frames.items():
        report = validate_dataset(df, name=name, near_threshold=near)
        path = _write_report(out_dir, name, report)
        log.info(f"{name}: report -> {path}")
        all_reports[name] = {"path": str(path), "summary": {
            "n_rows": report["schema"]["n_rows"],
            "schema_ok": report["schema"]["ok"],
            "exact_dups": report["duplicates"]["exact_duplicate_rows"],
            "near_dups": report["duplicates"]["near_duplicate_pairs"],
            "leakage": report["leakage"],
        }}

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    log.info(f"validation summary: {summary_path}")
    return all_reports


if __name__ == "__main__":
    cfg = load_data_config()
    run_all(cfg)
