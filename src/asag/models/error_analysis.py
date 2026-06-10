"""Phase 3 — error analysis (the reviewer-mandatory "what does it get wrong").

For each dataset, on its **headline split**, we take the per-item predictions the
GBM head already produces (reusing ``significance._collect_groups`` with the Phase
2D tuned params, so the analysis matches the reported headline) and characterize
the *structure* of the errors rather than just the aggregate metric:

* **ordinal** (QWK datasets) → exact / off-by-one / gross (|Δ|≥2) composition plus
  the signed bias (does the grader systematically over- or under-credit?) — the
  off-by-one rate is the number that tells a reviewer whether errors are benign
  adjacent-band disagreements or real failures.
* **classification** (macro-F1 datasets) → confusion matrix, per-class P/R/F1, and
  the most-confused label pairs.
* **regression** (Pearson datasets) → tolerance bands (|Δ|≤0.5 / ≤1.0 / >1.0),
  bias, MAE, RMSE on the original score scale.

If a Phase 2G neural prediction cache exists (``reports/phase2g/preds/<name>.json``)
the same ordinal/classification/regression breakdown is computed for the
cross-encoder too, giving a head-to-head error comparison.

    python -m asag.models.error_analysis [<name>...]   # -> reports/phase3/error_analysis.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from asag.config import DataConfig, ensure_dirs, load_data_config
from asag.models.data import load_bundle
from asag.models.significance import _collect_groups
from asag.models.tasks import REGISTRY, get_spec
from asag.utils.logging import get_logger
from asag.utils.seed import set_global_seed
from asag.xai.common import load_tuned_params

log = get_logger()


def _ordinal_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    e = np.rint(y_pred).astype(int) - np.rint(y_true).astype(int)
    n = int(e.size)
    if n == 0:
        return {"n": 0}
    return {"n": n,
            "exact": round(float(np.mean(e == 0)), 4),
            "off_by_one": round(float(np.mean(np.abs(e) == 1)), 4),
            "gross": round(float(np.mean(np.abs(e) >= 2)), 4),
            "mean_signed_error": round(float(np.mean(e)), 4),   # + = over-grading
            "mae": round(float(np.mean(np.abs(e))), 4)}


def _regression_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    e = np.asarray(y_pred, float) - np.asarray(y_true, float)
    n = int(e.size)
    if n == 0:
        return {"n": 0}
    a = np.abs(e)
    return {"n": n,
            "within_0.5": round(float(np.mean(a <= 0.5)), 4),
            "within_1.0": round(float(np.mean(a <= 1.0)), 4),
            "gross_gt_1.0": round(float(np.mean(a > 1.0)), 4),
            "bias": round(float(np.mean(e)), 4),
            "mae": round(float(np.mean(a)), 4),
            "rmse": round(float(np.sqrt(np.mean(e ** 2))), 4)}


def _classification_breakdown(y_true: np.ndarray, y_pred: np.ndarray,
                              inv_vocab: dict[int, str]) -> dict:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    yt = np.rint(y_true).astype(int)
    yp = np.rint(y_pred).astype(int)
    labels = sorted(set(yt.tolist()) | set(yp.tolist()))
    names = [inv_vocab.get(c, str(c)) for c in labels]
    cm = confusion_matrix(yt, yp, labels=labels)
    p, r, f, sup = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    per_class = {names[i]: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                            "f1": round(float(f[i]), 4), "support": int(sup[i])}
                 for i in range(len(labels))}
    # most-confused ordered (off-diagonal) pairs
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), names[i], names[j]))
    pairs.sort(reverse=True)
    top_confused = [{"true": t, "pred": pr, "count": c} for c, t, pr in pairs[:6]]
    return {"n": int(yt.size), "exact": round(float(np.mean(yt == yp)), 4),
            "labels": names, "confusion_matrix": cm.tolist(),
            "per_class": per_class, "top_confused": top_confused}


def _breakdown(task_type: str, yt: np.ndarray, yp: np.ndarray, inv_vocab: dict) -> dict:
    if task_type == "classification":
        return _classification_breakdown(yt, yp, inv_vocab)
    if task_type == "ordinal":
        return _ordinal_breakdown(yt, yp)
    return _regression_breakdown(yt, yp)


def _neural_headline_items(name: str, cfg: DataConfig, split: str) -> tuple | None:
    """Load cached neural (y_true, y_pred) for the headline split, if present."""
    p = cfg.paths.reports / "phase2g" / "preds" / f"{name}.json"
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    block = doc.get(split) or doc.get("cv")
    if not block or "y_true" not in block:
        return None
    return np.asarray(block["y_true"], float), np.asarray(block["y_pred"], float)


def analyze_dataset(name: str, cfg: DataConfig) -> dict | None:
    spec = get_spec(name)
    bundle = load_bundle(name, cfg, spec)
    if bundle is None:
        log.warning(f"{name}: features.parquet missing — skipping")
        return None
    params, source = load_tuned_params(name, cfg)
    groups, _clusters, split = _collect_groups(bundle, cfg, params)
    if not groups:
        return {"split": split, "status": "empty"}

    inv = {v: k for k, v in bundle.label_vocab.items()}
    yt = np.concatenate([g[0] for g in groups])
    gbm = np.concatenate([g[1] for g in groups])
    base = np.concatenate([g[2] for g in groups])
    out = {"task_type": spec.task_type, "metric": spec.headline, "split": split,
           "head_source": source,
           "gbm": _breakdown(spec.task_type, yt, gbm, inv),
           "baseline": _breakdown(spec.task_type, yt, base, inv)}

    if spec.per_prompt:
        out["per_prompt"] = _per_prompt_ordinal(bundle, cfg, params)

    neural = _neural_headline_items(name, cfg, split)
    if neural is not None:
        out["neural"] = _breakdown(spec.task_type, neural[0], neural[1], inv)
    log.info(f"{name}: error analysis done on {split} ({source})")
    return out


def _per_prompt_ordinal(bundle, cfg, params) -> dict:
    """ASAP-SAS: per-prompt exact/off-by-one/gross (each prompt its own rubric)."""
    from asag.models.evaluate import fit_predict_arrays
    spec, df = bundle.spec, bundle.df
    from asag.models.data import make_y
    finite = np.isfinite(make_y(df, bundle))
    split = spec.test_splits[-1]
    train = df[(df["split"] == "train") & finite]
    test = df[(df["split"] == split) & finite]
    res = {}
    for p in sorted(test["question_id"].astype(str).unique()):
        tr = train[train["question_id"].astype(str) == p]
        te = test[test["question_id"].astype(str) == p]
        if tr.empty or te.empty:
            continue
        yt, gbm, _ = fit_predict_arrays(tr, te, bundle, cfg, cfg.seed, params)
        res[p] = _ordinal_breakdown(yt, gbm)
    return res


def run_error_analysis(cfg: DataConfig | None = None, only: list[str] | None = None) -> dict:
    cfg = cfg or load_data_config()
    ensure_dirs(cfg)
    set_global_seed(cfg.seed)
    names = only or [n for n in REGISTRY if (cfg.paths.processed / n / "features.parquet").exists()]
    results = {n: r for n in names if (r := analyze_dataset(n, cfg)) is not None}
    if results:
        out_dir = cfg.paths.reports / "phase3"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "error_analysis.json").write_text(
            json.dumps({"datasets": results}, indent=2, default=str), encoding="utf-8")
        _write_figure(cfg, results)
        log.info(f"wrote {out_dir}/error_analysis.json")
    return results


def _write_figure(cfg: DataConfig, results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # composition bar: exact / near / gross per dataset (near = off-by-one or within_1.0)
    names, exact, near, gross = [], [], [], []
    for n, r in results.items():
        b = r.get("gbm", {})
        if r.get("task_type") == "ordinal":
            names.append(n); exact.append(b.get("exact", 0)); near.append(b.get("off_by_one", 0)); gross.append(b.get("gross", 0))
        elif r.get("task_type") == "regression":
            names.append(n); ex = b.get("within_0.5", 0); within1 = b.get("within_1.0", 0)
            exact.append(ex); near.append(max(0.0, within1 - ex)); gross.append(b.get("gross_gt_1.0", 0))
        else:  # classification: exact vs error (no "near")
            names.append(n); exact.append(b.get("exact", 0)); near.append(0.0); gross.append(1 - b.get("exact", 0))
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(names)), 5))
    ax.bar(x, exact, label="exact / within 0.5", color="#41ab5d")
    ax.bar(x, near, bottom=exact, label="off-by-one / within 1.0", color="#fdae6b")
    ax.bar(x, gross, bottom=np.array(exact) + np.array(near), label="gross (|Δ|≥2 / >1.0)", color="#cb181d")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("fraction of items"); ax.set_ylim(0, 1)
    ax.set_title("Phase 3 — error composition (GBM head, headline split)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(cfg.paths.figures / "phase3_error_composition.png", dpi=120)
    plt.close(fig)
    log.info(f"wrote {cfg.paths.figures / 'phase3_error_composition.png'}")


if __name__ == "__main__":
    import sys

    run_error_analysis(only=sys.argv[1:] or None)
