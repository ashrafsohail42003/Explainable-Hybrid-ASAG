"""Phase 3 — diagnosing the two generalization collapses (an honest finding).

Two results look like failures and a reviewer will demand an explanation rather
than a shrug:

1. **SAF cross-question cliff** — Pearson 0.91 on ``test_ua`` (unseen *answers* to
   *seen* questions) collapses to ~0.03 on ``test_uq`` (unseen *questions*). The
   hypothesis: the head keys on question-specific regularities, so it cannot
   transfer to a new prompt. We quantify it with the **between-question variance
   share** (η² of the gold score explained by ``question_id``) and a
   **question-mean shortcut** score — if simply predicting each item's
   *question-mean train score* already explains the seen-question performance, the
   model has little genuinely transferable signal.

2. **MIND-CA floor** — QWK ~0.12. The hypothesis: with no reference answer only the
   linguistic branch is populated (the semantic/rubric features are all-NaN), so
   the head has thin signal for a subjective construct. We quantify the **feature
   availability** (non-NaN share) and read it next to the error structure (mostly
   off-by-one, see error_analysis) — the floor is fine-grained indistinguishability,
   not catastrophic error.

A **length shortcut** probe (Pearson of answer length with the gold score) is run
for every dataset, since "longer answers score higher" is the classic ASAG
confound.

    python -m asag.models.collapse_analysis   # -> reports/phase3/collapse_analysis.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from asag.config import DataConfig, ensure_dirs, load_data_config
from asag.models.metrics import pearson
from asag.models.tasks import REGISTRY, get_spec
from asag.utils.logging import get_logger

log = get_logger()


def _eta_squared(score: pd.Series, group: pd.Series) -> float:
    """Share of score variance explained by the grouping (between-group SS / total)."""
    s = pd.to_numeric(score, errors="coerce")
    m = s.notna()
    s, g = s[m], group[m].astype(str)
    if s.size < 2 or g.nunique() < 2:
        return float("nan")
    grand = s.mean()
    ss_total = float(((s - grand) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    ss_between = float(sum(len(v) * (v.mean() - grand) ** 2 for _, v in s.groupby(g)))
    return round(ss_between / ss_total, 4)


def _question_mean_shortcut(df: pd.DataFrame) -> dict:
    """How well does 'predict the train question-mean' transfer to each split?

    Fit a question→mean map on train; apply to each split. On seen-question splits
    this is a strong proxy; on unseen-question splits the map misses (falls back to
    global mean → ~0 correlation), which is exactly the cliff we want to show.
    """
    s = pd.to_numeric(df["score"], errors="coerce")
    df = df.assign(_y=s)
    train = df[(df["split"] == "train") & df["_y"].notna()]
    if train.empty:
        return {}
    qmean = train.groupby(train["question_id"].astype(str))["_y"].mean()
    gmean = float(train["_y"].mean())
    out = {}
    for split in df["split"].unique():
        te = df[(df["split"] == split) & df["_y"].notna()]
        if te.empty or split == "train":
            continue
        pred = te["question_id"].astype(str).map(qmean).fillna(gmean).to_numpy()
        seen = te["question_id"].astype(str).isin(qmean.index).mean()
        out[split] = {"qmean_pearson": round(pearson(te["_y"].to_numpy(), pred), 4),
                      "frac_questions_seen": round(float(seen), 4), "n": int(len(te))}
    return out


def _length_shortcut(enc: pd.DataFrame) -> dict:
    s = pd.to_numeric(enc["score"], errors="coerce")
    length = enc["student_answer_enc"].fillna("").astype(str).str.len()
    m = s.notna()
    if m.sum() < 2:
        return {}
    return {"pearson_len_vs_score": round(pearson(s[m].to_numpy(), length[m].to_numpy()), 4),
            "median_answer_chars": int(length.median())}


def _feature_availability(name: str, cfg: DataConfig) -> dict:
    path = cfg.paths.processed / name / "features.parquet"
    if not path.exists():
        return {}
    from asag.models.data import KEY_COLUMNS
    df = pd.read_parquet(path)
    feats = [c for c in df.columns if c not in KEY_COLUMNS]
    nonnan = {c: float(df[c].notna().mean()) for c in feats}
    populated = [c for c, v in nonnan.items() if v > 0]
    return {"n_features": len(feats), "n_populated": len(populated),
            "frac_all_nan": round(1 - len(populated) / max(len(feats), 1), 4),
            "populated_columns": populated}


def analyze(name: str, cfg: DataConfig) -> dict:
    spec = get_spec(name)
    out: dict = {"task_type": spec.task_type}
    enc_p = cfg.paths.processed / name / "encoder.parquet"
    if enc_p.exists():
        enc = pd.read_parquet(enc_p).reset_index(drop=True)
        out["between_question_eta2"] = _eta_squared(enc["score"], enc["question_id"])
        out["length_shortcut"] = _length_shortcut(enc)
        if spec.protocol == "official_split" and spec.target == "score":
            out["question_mean_shortcut"] = _question_mean_shortcut(enc)
    out["feature_availability"] = _feature_availability(name, cfg)
    return out


def run_collapse_analysis(cfg: DataConfig | None = None, only: list[str] | None = None) -> dict:
    cfg = cfg or load_data_config()
    ensure_dirs(cfg)
    names = only or [n for n in REGISTRY if (cfg.paths.processed / n / "encoder.parquet").exists()]
    results = {n: analyze(n, cfg) for n in names}
    out_dir = cfg.paths.reports / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "collapse_analysis.json").write_text(
        json.dumps({"datasets": results}, indent=2, default=str), encoding="utf-8")
    for n, r in results.items():
        qm = r.get("question_mean_shortcut", {})
        log.info(f"{n}: eta2(question)={r.get('between_question_eta2')} "
                 f"len->score r={r.get('length_shortcut', {}).get('pearson_len_vs_score')} "
                 f"populated={r.get('feature_availability', {}).get('n_populated')}/"
                 f"{r.get('feature_availability', {}).get('n_features')}"
                 + (f" | qmean@uq={qm.get('test_uq', {}).get('qmean_pearson')}" if 'test_uq' in qm else ""))
    log.info(f"wrote {out_dir}/collapse_analysis.json")
    return results


if __name__ == "__main__":
    import sys

    run_collapse_analysis(only=sys.argv[1:] or None)
