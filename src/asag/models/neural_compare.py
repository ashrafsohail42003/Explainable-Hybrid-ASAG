"""Phase B — the three-way comparison that justifies a *hybrid* claim.

Once the DeBERTa cross-encoder has been run on Colab and ``neural_oof.parquet``
sits beside each ``features.parquet`` (so ``load_bundle`` concatenates the
``neural_*`` out-of-fold signals), this module reports, per dataset, the three
arms a reviewer needs side by side on the **same headline split**:

* **neural-only** — the raw DeBERTa held-out prediction read straight from the OOF
  columns (``neural_score`` for regression, ``neural_pred`` otherwise). This is the
  transformer baseline the GBM-only slices were missing.
* **feature-only** — the interpretable GBM head on the hand-engineered features
  with the ``neural_*`` columns dropped (the current system).
* **hybrid** — the same GBM head with the ``neural_*`` columns added (the fusion).

All three use the **same fixed, lightly-regularized head** (the Phase 3 ablation
head) so the only thing that moves between feature-only and hybrid is the presence
of the neural features — the Δ is attributable to the transformer, not to a head
or HPO difference (the tuned headline numbers live in ``reports/phase2d/``).

Two deltas drive the paper's claim:
* **fusion gain** = hybrid − feature-only  (does adding the transformer help?)
* **interpretability cost** = hybrid − neural-only  (what, if anything, does keeping
  the interpretable head cost vs the raw transformer?)

If the hybrid does not beat neural-only, the honest framing pivots to
"interpretability at near-parity cost" — the module reports the numbers either way.

    python -m asag.models.neural_compare [<name>...]   # datasets with neural_oof present
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from asag.config import DataConfig, LightGBMCfg, ensure_dirs, load_data_config
from asag.models.ablations import _ablation_head, _headline
from asag.models.data import Bundle, load_bundle, make_y
from asag.models.evaluate import _eval_kfold, _eval_official
from asag.models.fusion import LIGHTGBM_AVAILABLE
from asag.models.metrics import compute_metrics
from asag.models.tasks import REGISTRY, get_spec
from asag.utils.logging import get_logger
from asag.utils.seed import set_global_seed

log = get_logger()

PHASE_HYBRID_SCHEMA_VERSION = "hybrid.1"
_LOWER_IS_BETTER = {"rmse", "mae"}


def has_neural(bundle: Bundle) -> bool:
    return any(c.startswith("neural_") for c in bundle.feature_cols)


def neural_only_headline(bundle: Bundle) -> dict:
    """Raw DeBERTa headline from the OOF columns, on the dataset's headline split.

    Pure (no LightGBM): reads ``neural_score`` (regression) or ``neural_pred``
    (classification/ordinal) and scores it with the same metric, split, and
    per-prompt averaging the GBM headline uses. Returns ``{"status": "absent"}``
    when the OOF columns are not present yet.
    """
    spec, df = bundle.spec, bundle.df
    col = "neural_score" if spec.task_type == "regression" else "neural_pred"
    if col not in df.columns:
        return {"status": "absent"}
    y = make_y(df, bundle)
    pred = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(y) & np.isfinite(pred)

    def _metric(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return float("nan")
        return compute_metrics(y[mask], pred[mask], (spec.headline,)).get(spec.headline, float("nan"))

    if spec.protocol == "kfold":
        split, val = "cv", _metric(finite & (df["fold"].to_numpy() >= 0))
    else:
        split = spec.test_splits[-1]
        in_split = finite & (df["split"].to_numpy() == split)
        if spec.per_prompt:
            qid = df["question_id"].astype(str).to_numpy()
            vals = [m for p in np.unique(qid[in_split])
                    if np.isfinite(m := _metric(in_split & (qid == p)))]
            val = float(np.mean(vals)) if vals else float("nan")
        else:
            val = _metric(in_split)
    return {"status": "ok", "split": split,
            "headline": {"metric": spec.headline, "mean": round(float(val), 4)
                         if np.isfinite(val) else None}}


def _gbm_headline(bundle: Bundle, cfg: DataConfig, head: LightGBMCfg) -> dict:
    spec = bundle.spec
    evals = (_eval_official(bundle, cfg, head) if spec.protocol == "official_split"
             else _eval_kfold(bundle, cfg, head))
    h = _headline(evals, spec)
    return {"metric": spec.headline, "mean": h.get("mean"), "std": h.get("std")}


def _delta(a: float | None, b: float | None, headline: str) -> float | None:
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return None
    sign = -1.0 if headline in _LOWER_IS_BETTER else 1.0
    return round(float(sign * (a - b)), 4)


def compare_dataset(name: str, cfg: DataConfig, head: LightGBMCfg) -> dict | None:
    spec = get_spec(name)
    bundle = load_bundle(name, cfg, spec)
    if bundle is None:
        log.warning(f"{name}: features.parquet missing — skipping")
        return None
    if not has_neural(bundle):
        log.info(f"{name}: no neural_oof.parquet yet — run the Colab notebook first")
        return {"status": "no_neural", "metric": spec.headline}

    feat_cols = [c for c in bundle.feature_cols if not c.startswith("neural_")]
    feature_only = _gbm_headline(replace(bundle, feature_cols=feat_cols), cfg, head)
    hybrid = _gbm_headline(bundle, cfg, head)
    neural = neural_only_headline(bundle).get("headline", {})

    fusion_gain = _delta(hybrid.get("mean"), feature_only.get("mean"), spec.headline)
    interp_cost = _delta(hybrid.get("mean"), neural.get("mean"), spec.headline)
    log.info(f"{name}: neural={neural.get('mean')} feature={feature_only.get('mean')} "
             f"hybrid={hybrid.get('mean')} | fusion_gain={fusion_gain} interp_cost={interp_cost}")
    return {"status": "ok", "metric": spec.headline,
            "headline_split": spec.test_splits[-1] if spec.protocol == "official_split" else "cv",
            "neural_only": neural, "feature_only": feature_only, "hybrid": hybrid,
            "fusion_gain": fusion_gain, "interpretability_cost": interp_cost}


def run_compare(cfg: DataConfig | None = None, only: list[str] | None = None) -> dict:
    cfg = cfg or load_data_config()
    ensure_dirs(cfg)
    set_global_seed(cfg.seed)
    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError("lightgbm is not installed; run `uv pip install lightgbm`")
    head = _ablation_head(cfg)
    names = only or [n for n in REGISTRY if (cfg.paths.processed / n / "features.parquet").exists()]
    results = {n: r for n in names if (r := compare_dataset(n, cfg, head)) is not None}
    ready = {n: r for n, r in results.items() if r.get("status") == "ok"}
    if ready:
        _write_reports(cfg, results, head)
        _write_figure(cfg, ready)
    else:
        log.warning("no dataset has neural_oof.parquet yet — nothing to compare")
    return results


# ----------------------------- reporting ---------------------------------

def _write_reports(cfg: DataConfig, results: dict, head: LightGBMCfg) -> None:
    out_dir = cfg.paths.reports / "phase_hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "three_way.json").write_text(json.dumps(
        {"schema_version": PHASE_HYBRID_SCHEMA_VERSION, "seeds": list(cfg.model.seeds),
         "head": head.model_dump(), "datasets": results}, indent=2, default=str),
        encoding="utf-8")

    lines = ["# Phase B — three-way comparison: neural-only / feature-only / hybrid\n",
             "> One fixed regularized head for feature-only and hybrid (the Δ is the",
             "> neural features). Tuned headline numbers live in `reports/phase2d/`.\n",
             "| Dataset | Metric | Neural-only | Feature-only | Hybrid | Fusion gain | Interp. cost |",
             "|---|---|---|---|---|---|---|"]

    def f(x):
        return "—" if x is None else f"{x:.4f}"
    for n, r in results.items():
        if r.get("status") != "ok":
            lines.append(f"| {n} | {r.get('metric','')} | _neural_oof not present_ | | | | |")
            continue
        lines.append(f"| {n} | {r['metric']} | {f(r['neural_only'].get('mean'))} | "
                     f"{f(r['feature_only'].get('mean'))} | {f(r['hybrid'].get('mean'))} | "
                     f"{f(r['fusion_gain'])} | {f(r['interpretability_cost'])} |")
    (out_dir / "three_way.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"wrote {out_dir}/ three_way.json, three_way.md")


def _write_figure(cfg: DataConfig, ready: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(ready.keys())
    arms = [("neural_only", "neural-only", "#9e9ac8"),
            ("feature_only", "feature-only", "#74c476"),
            ("hybrid", "hybrid", "#238b45")]
    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 1.9 * len(names)), 5))
    for k, (key, lab, color) in enumerate(arms):
        means = [ready[n][key].get("mean", np.nan) for n in names]
        ax.bar(x + (k - 1) * w, means, w, label=lab, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({ready[n]['metric']})" for n in names], fontsize=8)
    ax.set_ylabel("headline metric")
    ax.set_title("Phase B — neural-only vs feature-only vs hybrid (fixed head)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.paths.figures / "phase_hybrid_three_way.png", dpi=120)
    plt.close(fig)
    log.info(f"wrote {cfg.paths.figures / 'phase_hybrid_three_way.png'}")


if __name__ == "__main__":
    import sys
    run_compare(only=sys.argv[1:] or None)
