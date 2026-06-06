"""Phase 2D CLI — rigorous training: Optuna HPO + paired-bootstrap + IAA ceiling.

Pipeline per dataset:

1. ``hpo.tune_dataset`` — Optuna search (dev / inner-CV objective) → tuned ``LightGBMCfg``.
2. ``evaluate.evaluate_dataset(..., head_params=tuned)`` — the *unchanged* Phase 2C
   multi-seed protocol, now with the tuned head (subsample/colsample < 1 give an
   honest, non-zero seed std).
3. ``significance.paired_bootstrap`` — Δheadline (head − baseline) on the headline
   split, with a CI and one-sided p-value.
4. ``ceiling.ceiling_for`` — the ASAP-SAS human (inter-annotator) QWK.

Outputs land in ``reports/phase2d/`` (hpo/results/significance/ceiling JSON + a
flattened results.csv) plus three figures. Phase 2C is left untouched: passing
``head_params=None`` anywhere reproduces it byte-for-byte.

Usage::

    python -m asag.models.train2d                 # all datasets with features
    python -m asag.models.train2d mohler asap_sas # a subset (stay under the C: ceiling)
"""

from __future__ import annotations

import json

import numpy as np

from asag.config import DataConfig, ensure_dirs, load_data_config
from asag.models.ceiling import ceiling_for
from asag.models.evaluate import _flatten_rows, evaluate_dataset
from asag.models.fusion import LIGHTGBM_AVAILABLE
from asag.models.hpo import tune_dataset
from asag.models.significance import paired_bootstrap
from asag.models.data import load_bundle
from asag.models.tasks import REGISTRY, get_spec
from asag.utils.logging import get_logger
from asag.utils.seed import set_global_seed

log = get_logger()

PHASE2D_SCHEMA_VERSION = "2d.1"


def _phase2c_headline(cfg: DataConfig) -> dict[str, float]:
    """Best-effort read of the Phase 2C default-head headline means (for the figure)."""
    path = cfg.paths.reports / "phase2c" / "results.json"
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {n: d.get("headline", {}).get("gbm", {}).get("mean", np.nan)
            for n, d in doc.get("datasets", {}).items()}


def run_phase2d(cfg: DataConfig | None = None, only: list[str] | None = None) -> dict:
    cfg = cfg or load_data_config()
    ensure_dirs(cfg)
    set_global_seed(cfg.seed)
    if not cfg.model.enabled:
        log.warning("model.enabled is false — nothing to do")
        return {}
    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError("lightgbm is not installed; run `uv pip install lightgbm`")

    names = only or [n for n in REGISTRY if (cfg.paths.processed / n / "features.parquet").exists()]
    hpo_summaries: dict[str, dict] = {}
    results: dict[str, dict] = {}
    significance: dict[str, dict] = {}
    ceilings: dict[str, dict] = {}

    for name in names:
        spec = get_spec(name)
        bundle = load_bundle(name, cfg, spec)
        if bundle is None:
            log.warning(f"{name}: features.parquet missing — run `make features`; skipping")
            continue

        tuned, summary = tune_dataset(bundle, cfg)
        hpo_summaries[name] = summary

        res = evaluate_dataset(name, cfg, head_params=tuned)
        if res is None:
            continue
        res["lightgbm_tuned"] = tuned.model_dump()
        res["hpo_validation"] = summary.get("validation")
        results[name] = res

        if cfg.model.significance.enabled:
            significance[name] = paired_bootstrap(bundle, cfg, tuned)
        ceilings[name] = ceiling_for(name, cfg)

    if results:
        _write_reports(cfg, hpo_summaries, results, significance, ceilings)
        _write_figures(cfg, results, significance, ceilings)
    return results


# ----------------------------- reporting ---------------------------------

def _write_reports(cfg: DataConfig, hpo_summaries: dict, results: dict,
                   significance: dict, ceilings: dict) -> None:
    out_dir = cfg.paths.reports / "phase2d"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "hpo.json").write_text(json.dumps(
        {"schema_version": PHASE2D_SCHEMA_VERSION,
         "n_trials": cfg.model.hpo.n_trials,
         "datasets": hpo_summaries}, indent=2, default=str), encoding="utf-8")

    doc = {"schema_version": PHASE2D_SCHEMA_VERSION,
           "fusion_head": cfg.model.fusion_head,
           "seeds": list(cfg.model.seeds),
           "lightgbm_defaults": cfg.model.lightgbm.model_dump(),
           "datasets": results}
    (out_dir / "results.json").write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")

    import pandas as pd
    rows = [r for name, res in results.items() for r in _flatten_rows(name, res)]
    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    (out_dir / "significance.json").write_text(json.dumps(
        {"schema_version": PHASE2D_SCHEMA_VERSION, "datasets": significance},
        indent=2, default=str), encoding="utf-8")
    (out_dir / "ceiling.json").write_text(json.dumps(
        {"schema_version": PHASE2D_SCHEMA_VERSION, "datasets": ceilings},
        indent=2, default=str), encoding="utf-8")
    log.info(f"wrote {out_dir}/ hpo.json, results.json, results.csv, significance.json, ceiling.json")


def _write_figures(cfg: DataConfig, results: dict, significance: dict, ceilings: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = cfg.paths.figures
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    c2 = _phase2c_headline(cfg)

    # 1) tuned GBM vs Phase 2C default vs baseline (headline per dataset).
    tuned_m = [results[n]["headline"]["gbm"].get("mean", np.nan) for n in names]
    tuned_s = [results[n]["headline"]["gbm"].get("std", 0.0) or 0.0 for n in names]
    def2c_m = [c2.get(n, np.nan) for n in names]
    base_m = [results[n]["headline"]["baseline"].get("mean", np.nan) for n in names]
    labels = [f"{n}\n({results[n]['headline']['metric']}@{results[n]['headline']['split']})" for n in names]
    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 1.9 * len(names)), 5))
    ax.bar(x - w, tuned_m, w, yerr=tuned_s, capsize=4, label="GBM tuned (2D)", color="#238b45")
    ax.bar(x, def2c_m, w, label="GBM default (2C)", color="#74c476")
    ax.bar(x + w, base_m, w, label="naive baseline", color="#bdbdbd")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("headline metric"); ax.set_title("Phase 2D — tuned head vs Phase 2C default vs baseline")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "phase2d_tuned_vs_2c.png", dpi=120); plt.close(fig)

    # 2) Δheadline with bootstrap CI per dataset.
    sig_names = [n for n in names if significance.get(n, {}).get("status") == "ok"]
    if sig_names:
        deltas = [significance[n]["delta_observed"] for n in sig_names]
        lo = [significance[n]["delta_observed"] - significance[n]["ci_lo"] for n in sig_names]
        hi = [significance[n]["ci_hi"] - significance[n]["delta_observed"] for n in sig_names]
        xs = np.arange(len(sig_names))
        colors = ["#238b45" if significance[n]["significant"] else "#fb6a4a" for n in sig_names]
        fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(sig_names)), 5))
        ax.bar(xs, deltas, 0.5, yerr=[lo, hi], capsize=5, color=colors)
        ax.axhline(0.0, color="#333", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{n}\n(Δ{significance[n]['metric']}, p={significance[n]['p_value']:.3f})"
                            for n in sig_names], fontsize=8)
        ax.set_ylabel("Δ headline (head − baseline)")
        ax.set_title(f"Phase 2D — paired bootstrap ({cfg.model.significance.n_boot} resamples, "
                     f"{int(cfg.model.significance.ci * 100)}% CI)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(fig_dir / "phase2d_significance.png", dpi=120); plt.close(fig)

    # 3) ASAP-SAS model QWK@test_ua vs human (IAA) ceiling, per prompt + macro.
    cl = ceilings.get("asap_sas", {})
    if cl.get("status") == "ok" and "asap_sas" in results:
        ev = results["asap_sas"]["evaluations"].get("test_ua", {})
        model_macro = ev.get("gbm", {}).get("qwk", {}).get("mean", np.nan)
        per_prompt = results["asap_sas"].get("evaluations", {}).get("test_ua", {}).get("per_prompt", {})
        prompts = sorted(cl["per_prompt"].keys())
        ceil_vals = [cl["per_prompt"][p]["qwk"] for p in prompts]
        model_vals = [per_prompt.get(p, {}).get("qwk", {}).get("mean", np.nan) for p in prompts]
        cats = prompts + ["macro"]
        ceil_all = ceil_vals + [cl["macro_qwk"]]
        model_all = model_vals + [model_macro]
        xs = np.arange(len(cats)); w = 0.38
        fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(cats)), 5))
        ax.bar(xs - w / 2, model_all, w, label="GBM tuned (test QWK)", color="#238b45")
        ax.bar(xs + w / 2, ceil_all, w, label="human IAA ceiling (train+dev)", color="#9ecae1")
        ax.set_xticks(xs); ax.set_xticklabels(cats, fontsize=9)
        ax.set_ylabel("QWK"); ax.set_ylim(0, 1)
        ax.set_title("Phase 2D — ASAP-SAS model QWK vs human ceiling (Score1 vs Score2)")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(fig_dir / "phase2d_ceiling.png", dpi=120); plt.close(fig)
    log.info(f"wrote Phase 2D figures to {fig_dir}")


if __name__ == "__main__":
    import sys

    run_phase2d(only=sys.argv[1:] or None)
