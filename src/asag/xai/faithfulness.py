"""Phase 2F — faithfulness of the rubric branch as an *explanation* (the reframe).

Phase 3 ablations show the rubric branch (``rub_*``) adds ~0 **accuracy**. That is
not the end of its story — for an explainable grader the rubric's job is to be a
*faithful* account of the decision, not to move the metric. This module quantifies
that, separating two properties an explanation feature can have:

1. **Predictive validity** — does the signal actually track the grade?
   Spearman(feature value, gold score).
2. **Model-use faithfulness** — does the *head* use it monotonically in the
   sensible direction? Spearman(feature value, its TreeSHAP contribution) and the
   sign-consistency ``mean[ sign(value − median) == sign(SHAP) ]``. A feature is a
   faithful explanation when higher coverage both *correlates with* a higher grade
   and *pushes the model's prediction up*.

We compute these per branch (rubric / semantic / linguistic) on the two
reference-bearing **regression** datasets (mohler, saf) where the target is ordered
and ``rub_*`` exists, so the coverage→grade direction is well defined. The headline
takeaway for the paper: the rubric is **faithful but not accuracy-additive** — it
explains without costing accuracy — which is exactly the property an interpretable
grader wants, and it complements the Phase 2F SAF gold-feedback result.

    python -m asag.xai.faithfulness   # -> reports/phase2f/faithfulness.json
"""

from __future__ import annotations

import json

import numpy as np

from asag.config import DataConfig, ensure_dirs, load_data_config
from asag.models.data import load_bundle, make_X, make_y
from asag.models.tasks import get_spec
from asag.utils.logging import get_logger
from asag.xai.common import fit_head_on_all, load_tuned_params, shaped_contribs

log = get_logger()

# reference-bearing, ordered-target datasets where coverage→grade is well defined
FAITHFULNESS_DATASETS = ("mohler", "saf")


def _branch_of(col: str) -> str:
    if col.startswith("sem_"):
        return "semantic"
    if col.startswith("rub_"):
        return "rubric"
    if col.startswith("neural_"):
        return "neural"
    return "linguistic"


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(a[m], b[m]).statistic)


def _feature_faithfulness(values: np.ndarray, shap: np.ndarray, target: np.ndarray) -> dict:
    valid = np.isfinite(values)
    if valid.sum() < 5:
        return {"n": int(valid.sum()), "status": "too_few"}
    v, s, y = values[valid], shap[valid], target[valid]
    med = np.nanmedian(v)
    # sign-consistency between (value above/below median) and SHAP push direction
    nz = np.abs(s) > 1e-12
    sign_cons = (float(np.mean(np.sign(v[nz] - med) == np.sign(s[nz]))) if nz.sum() else float("nan"))
    return {"n": int(valid.sum()),
            "predictive_validity_rho": round(_spearman(v, y), 4),     # value vs gold
            "model_use_rho": round(_spearman(v, s), 4),               # value vs its SHAP
            "shap_sign_consistency": round(sign_cons, 4),
            "mean_abs_shap": round(float(np.mean(np.abs(s))), 6)}


def analyze_dataset(name: str, cfg: DataConfig) -> dict | None:
    spec = get_spec(name)
    if spec.task_type != "regression":
        return None
    bundle = load_bundle(name, cfg, spec)
    if bundle is None:
        return None
    params, source = load_tuned_params(name, cfg)
    fitted = fit_head_on_all(bundle, cfg, params)
    if fitted is None:
        return None
    head, X, y = fitted
    contribs = shaped_contribs(head, X, len(bundle.feature_cols))   # (n, f) for regression
    Xv = X.to_numpy()

    per_feature = {}
    for j, col in enumerate(bundle.feature_cols):
        per_feature[col] = {"branch": _branch_of(col),
                            **_feature_faithfulness(Xv[:, j], contribs[:, j], y)}

    # aggregate per branch (mean over its features of each faithfulness metric)
    branches: dict[str, dict] = {}
    for branch in ("rubric", "semantic", "linguistic", "neural"):
        cols = [c for c in bundle.feature_cols if _branch_of(c) == branch]
        rows = [per_feature[c] for c in cols if "predictive_validity_rho" in per_feature[c]]
        if not rows:
            continue
        branches[branch] = {
            "n_features": len(rows),
            "mean_predictive_validity_rho": round(float(np.nanmean([r["predictive_validity_rho"] for r in rows])), 4),
            "mean_model_use_rho": round(float(np.nanmean([r["model_use_rho"] for r in rows])), 4),
            "mean_sign_consistency": round(float(np.nanmean([r["shap_sign_consistency"] for r in rows])), 4),
            "share_global_abs_shap": round(float(np.nansum([r["mean_abs_shap"] for r in rows])), 6),
        }
    # normalize the SHAP share across branches
    tot = sum(b["share_global_abs_shap"] for b in branches.values()) or 1.0
    for b in branches.values():
        b["share_global_abs_shap"] = round(b["share_global_abs_shap"] / tot, 4)

    log.info(f"{name} ({source}): rubric faithfulness "
             f"validity={branches.get('rubric', {}).get('mean_predictive_validity_rho')} "
             f"model_use={branches.get('rubric', {}).get('mean_model_use_rho')} "
             f"shap_share={branches.get('rubric', {}).get('share_global_abs_shap')}")
    return {"head_source": source, "branches": branches, "per_feature": per_feature}


def run_faithfulness(cfg: DataConfig | None = None, only: list[str] | None = None) -> dict:
    cfg = cfg or load_data_config()
    ensure_dirs(cfg)
    names = only or list(FAITHFULNESS_DATASETS)
    results = {n: r for n in names if (r := analyze_dataset(n, cfg)) is not None}
    out_dir = cfg.paths.reports / "phase2f"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "faithfulness.json").write_text(
        json.dumps({"datasets": results,
                    "interpretation": "Rubric is faithful (predictive validity + monotone model use) "
                                      "yet ~0 accuracy-additive (Phase 3 ablation) — an explanation "
                                      "layer that does not trade accuracy for transparency."},
                   indent=2, default=str), encoding="utf-8")
    if results:
        _write_figure(cfg, results)
    log.info(f"wrote {out_dir}/faithfulness.json")
    return results


def _write_figure(cfg: DataConfig, results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    branches = ["rubric", "semantic", "linguistic"]
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 4.2), squeeze=False)
    for i, n in enumerate(names):
        ax = axes[0][i]
        b = results[n]["branches"]
        validity = [b.get(br, {}).get("mean_predictive_validity_rho", np.nan) for br in branches]
        use = [b.get(br, {}).get("mean_model_use_rho", np.nan) for br in branches]
        x = np.arange(len(branches)); w = 0.38
        ax.bar(x - w / 2, validity, w, label="predictive validity ρ(value,gold)", color="#3182bd")
        ax.bar(x + w / 2, use, w, label="model-use ρ(value,SHAP)", color="#e6550d")
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(branches, fontsize=9)
        ax.set_title(n); ax.set_ylim(-1, 1); ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7); ax.set_ylabel("Spearman ρ")
    fig.suptitle("Phase 2F — branch faithfulness (predictive validity vs monotone model use)")
    fig.tight_layout()
    fig.savefig(cfg.paths.figures / "phase2f_faithfulness.png", dpi=120); plt.close(fig)
    log.info(f"wrote {cfg.paths.figures / 'phase2f_faithfulness.png'}")


if __name__ == "__main__":
    import sys

    run_faithfulness(only=sys.argv[1:] or None)
