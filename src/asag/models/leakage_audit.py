"""Phase 4 — leakage / shortcut audit (the diagnosis behind the eval fix).

For every dataset, quantifies the three ways a head can look good without grading:

* **question memorization** — `between_question_eta2` (fraction of target variance
  that is between questions) and the per-question shortcut metric under grouped vs
  stratified folds (it should die when questions are held out).
* **length shortcut** — Pearson(answer length, score).
* **stratified→grouped inflation** — the same head's headline under the legacy
  score-stratified k-fold vs grouped-by-question k-fold (k-fold datasets only).

Pure-ish: reads the existing ``features.parquet`` (no SBERT). Writes
``reports/phase4_audit/leakage_audit.json``. ``python -m asag.models.leakage_audit``.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from asag.config import DataConfig, load_data_config
from asag.data.splits import make_grouped_kfold, make_stratified_kfold
from asag.models.baselines import question_shortcut_predict
from asag.models.data import load_bundle, make_X, make_y
from asag.models.fusion import LIGHTGBM_AVAILABLE, LgbmFusionHead
from asag.models.metrics import compute_metrics
from asag.models.tasks import REGISTRY, get_spec
from asag.utils.logging import get_logger
from asag.utils.seed import set_global_seed

log = get_logger()


def between_question_eta2(df: pd.DataFrame) -> float:
    """η² of score across question groups (between-group SS / total SS)."""
    s = pd.to_numeric(df["score"], errors="coerce")
    g = df["question_id"].astype(str)
    d = pd.DataFrame({"s": s, "g": g}).dropna()
    if d["s"].nunique() <= 1 or d["g"].nunique() <= 1:
        return float("nan")
    grand = d["s"].mean()
    ss_tot = float(((d["s"] - grand) ** 2).sum())
    ss_bet = float(d.groupby("g")["s"].apply(lambda x: len(x) * (x.mean() - grand) ** 2).sum())
    return round(ss_bet / ss_tot, 4) if ss_tot > 0 else float("nan")


def length_shortcut(df: pd.DataFrame) -> float:
    s = pd.to_numeric(df["score"], errors="coerce")
    ln = df["student_answer_feat"].fillna("").str.len() if "student_answer_feat" in df \
        else df.get("len_student_chars")
    d = pd.DataFrame({"s": s, "l": pd.to_numeric(ln, errors="coerce")}).dropna()
    if len(d) < 3 or d["s"].nunique() <= 1 or d["l"].nunique() <= 1:
        return float("nan")
    return round(float(np.corrcoef(d["l"], d["s"])[0, 1]), 4)


def _pooled(bundle, fold: np.ndarray, cfg, head_cfg) -> tuple[float, float]:
    """Pooled-OOF (gbm_headline, qshortcut_headline) over a fold assignment, seed 42."""
    spec, df = bundle.spec, bundle.df
    y = make_y(df, bundle)
    finite = np.isfinite(y)
    yt, gp, st_t, st_p = [], [], [], []
    for f in sorted(int(x) for x in np.unique(fold) if int(x) >= 0):
        tr = df[(fold != f) & (fold >= 0) & finite]
        te = df[(fold == f) & finite]
        if tr.empty or te.empty:
            continue
        Xtr, ytr = make_X(tr, bundle.feature_cols), make_y(tr, bundle)
        Xte, yte = make_X(te, bundle.feature_cols), make_y(te, bundle)
        set_global_seed(42)
        head = LgbmFusionHead(spec.task_type, head_cfg, 42).fit(Xtr, ytr)
        yt.append(yte); gp.append(head.predict(Xte))
        st_t.append(yte)
        st_p.append(question_shortcut_predict(ytr, tr["question_id"], te["question_id"], spec.task_type))
    if not yt:
        return float("nan"), float("nan")
    h = spec.headline
    return (round(compute_metrics(np.concatenate(yt), np.concatenate(gp), spec.metrics)[h], 4),
            round(compute_metrics(np.concatenate(st_t), np.concatenate(st_p), spec.metrics)[h], 4))


def run_all(cfg: DataConfig | None = None, names: list[str] | None = None) -> dict:
    cfg = cfg or load_data_config()
    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError("lightgbm is not installed; run `uv pip install lightgbm`")
    names = names or [n for n in REGISTRY
                      if (cfg.paths.processed / n / "features.parquet").exists()]
    head_cfg = cfg.model.lightgbm
    out: dict[str, dict] = {}
    for name in names:
        spec = get_spec(name)
        bundle = load_bundle(name, cfg, spec)
        df = bundle.df
        entry = {
            "task_type": spec.task_type,
            "headline": spec.headline,
            "between_question_eta2": between_question_eta2(df),
            "length_vs_score_pearson": length_shortcut(df),
        }
        if spec.protocol == "kfold":
            grouped = df["fold"].to_numpy()
            strat = make_stratified_kfold(df, k=cfg.splits.cv_k_folds, seed=cfg.seed,
                                          stratify_on=cfg.splits.stratify_on).to_numpy()
            g_gbm, g_sc = _pooled(bundle, grouped, cfg, head_cfg)
            s_gbm, s_sc = _pooled(bundle, strat, cfg, head_cfg)
            entry["gbm_stratified"] = s_gbm
            entry["gbm_grouped"] = g_gbm
            entry["gbm_inflation"] = round(s_gbm - g_gbm, 4)
            entry["qshortcut_stratified"] = s_sc
            entry["qshortcut_grouped"] = g_sc
            log.info(f"{name}: inflation={entry['gbm_inflation']:+.4f} "
                     f"(strat {s_gbm:.4f} → grouped {g_gbm:.4f}); eta2={entry['between_question_eta2']}")
        else:
            log.info(f"{name}: eta2={entry['between_question_eta2']} "
                     f"len↔score={entry['length_vs_score_pearson']}")
        out[name] = entry

    doc = {"head": head_cfg.model_dump(), "datasets": out}
    dst = cfg.paths.reports / "phase4_audit"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "leakage_audit.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    log.info(f"wrote {dst / 'leakage_audit.json'}")
    return doc


if __name__ == "__main__":
    import sys
    run_all(names=sys.argv[1:] or None)
