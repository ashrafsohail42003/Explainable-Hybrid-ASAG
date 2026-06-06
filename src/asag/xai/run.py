"""Phase 2F CLI — explainability bundle: SHAP + concept coverage + SAF validation.

    python -m asag.xai.run                 # all datasets with features
    python -m asag.xai.run saf mohler      # a subset

Writes ``reports/phase2f/{shap,concept_attribution,saf_validation}.json`` plus
three figures. GBM-only (no torch); SHAP is LightGBM-native (no ``shap`` dep). The
SBERT encoder is loaded once and shared across concept attribution + SAF validation.
"""

from __future__ import annotations

import json

import numpy as np

from asag.config import DataConfig, ensure_dirs, load_data_config
from asag.models.data import load_bundle
from asag.models.fusion import LIGHTGBM_AVAILABLE
from asag.models.tasks import REGISTRY, get_spec
from asag.utils.logging import get_logger
from asag.utils.seed import set_global_seed
from asag.xai import XAI_SCHEMA_VERSION
from asag.xai.concept_attribution import attribute_examples
from asag.xai.saf_validation import validate_saf
from asag.xai.shap_explain import explain_dataset

log = get_logger()


def run_xai(cfg: DataConfig | None = None, only: list[str] | None = None) -> dict:
    cfg = cfg or load_data_config()
    ensure_dirs(cfg)
    set_global_seed(cfg.seed)
    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError("lightgbm is not installed; run `uv pip install lightgbm`")

    names = only or [n for n in REGISTRY if (cfg.paths.processed / n / "features.parquet").exists()]

    shap_out: dict[str, dict] = {}
    concept_out: dict[str, dict] = {}
    for name in names:
        bundle = load_bundle(name, cfg, get_spec(name))
        if bundle is None:
            log.warning(f"{name}: features.parquet missing — skipping")
            continue
        res = explain_dataset(name, bundle, cfg)
        if res is not None:
            shap_out[name] = res
            top = res["global_importance"][0]
            log.info(f"{name}: SHAP top feature = {top['feature']} ({top['mean_abs_shap']}); "
                     f"SHAP↔gain ρ={res['shap_vs_gain_spearman']}")

    # Concept attribution + SAF validation share one SBERT encoder + spaCy pipe.
    encoder, nlp = None, None
    if any((cfg.paths.processed / n / "encoder.parquet").exists() for n in names) or "saf" in names:
        from asag.features.semantic import SbertEncoder
        from asag.features.text_utils import load_feature_nlp
        encoder = SbertEncoder(cfg.features.sbert_model,
                               batch_size=cfg.features.semantic.batch_size, normalize=True)
        nlp = load_feature_nlp(cfg.features.ner.spacy_model)

    for name in names:
        att = attribute_examples(name, cfg, encoder, nlp)
        concept_out[name] = att
        if att.get("status") == "ok":
            covs = [e["coverage_fraction"] for e in att["examples"] if e["coverage_fraction"] is not None]
            log.info(f"{name}: concept attribution on {att['n_examples']} examples "
                     f"(coverage {min(covs):.2f}–{max(covs):.2f})" if covs else f"{name}: concept attribution")

    saf = validate_saf(cfg, encoder, nlp) if (only is None or "saf" in names) else {"status": "skipped"}
    if saf.get("status") == "ok":
        cov = saf["signals"]["rub_coverage_at_tau"]
        log.info(f"saf novelty: coverage by verdict {cov['by_class']} "
                 f"(ρ={cov['spearman_vs_verdict']}, AUC={cov['auc_correct_vs_rest']})")

    if shap_out or concept_out or saf.get("status") == "ok":
        _write_reports(cfg, shap_out, concept_out, saf)
        _write_figures(cfg, shap_out, concept_out, saf)
    return {"shap": shap_out, "concept": concept_out, "saf": saf}


# ----------------------------- reporting ---------------------------------

def _write_reports(cfg: DataConfig, shap_out: dict, concept_out: dict, saf: dict) -> None:
    out = cfg.paths.reports / "phase2f"
    out.mkdir(parents=True, exist_ok=True)
    for fname, payload in (("shap.json", shap_out),
                           ("concept_attribution.json", concept_out),
                           ("saf_validation.json", saf)):
        (out / fname).write_text(json.dumps(
            {"schema_version": XAI_SCHEMA_VERSION,
             ("datasets" if fname != "saf_validation.json" else "result"): payload},
            indent=2, default=str), encoding="utf-8")
    log.info(f"wrote {out}/ shap.json, concept_attribution.json, saf_validation.json")


def _write_figures(cfg: DataConfig, shap_out: dict, concept_out: dict, saf: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = cfg.paths.figures
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) SHAP global importance (top 8) — small multiples over datasets.
    have = [(n, r) for n, r in shap_out.items() if r.get("global_importance")]
    if have:
        cols = min(3, len(have)); rows = int(np.ceil(len(have) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.3 * rows), squeeze=False)
        for i, (n, r) in enumerate(have):
            ax = axes[i // cols][i % cols]
            top = r["global_importance"][:8][::-1]
            ax.barh([d["feature"] for d in top], [d["mean_abs_shap"] for d in top], color="#756bb1")
            ax.set_title(f"{n} (ρ_gain={r['shap_vs_gain_spearman']})", fontsize=9)
            ax.tick_params(labelsize=7)
        for j in range(len(have), rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.suptitle("Phase 2F — SHAP global importance (mean |SHAP|, top 8)", fontsize=11)
        fig.tight_layout(); fig.savefig(fig_dir / "phase2f_shap_global.png", dpi=120); plt.close(fig)

    # 2) SAF novelty — interpretable signal mean by human verdict (monotone = aligned).
    if saf.get("status") == "ok":
        order = ["Incorrect", "Partially correct", "Correct"]
        sigs = list(saf["signals"].keys())
        x = np.arange(len(order)); w = 0.2
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, s in enumerate(sigs):
            bc = saf["signals"][s]["by_class"]
            ax.bar(x + (k - 1.5) * w, [bc.get(o, np.nan) for o in order], w,
                   label=f"{s} (ρ={saf['signals'][s]['spearman_vs_verdict']}, AUC={saf['signals'][s]['auc_correct_vs_rest']})")
        ax.set_xticks(x); ax.set_xticklabels(order)
        ax.set_ylabel("mean signal value"); ax.set_title("Phase 2F — SAF: interpretable signals vs human gold verdict")
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(fig_dir / "phase2f_saf_validation.png", dpi=120); plt.close(fig)

    # 3) Concept-coverage example — one SAF answer, per-concept covered/missed.
    saf_ex = concept_out.get("saf", {})
    if saf_ex.get("status") == "ok" and saf_ex["examples"]:
        ex = max(saf_ex["examples"], key=lambda e: e["n_concepts"])
        concepts = ex["concepts"]
        labels = [f"c{i+1}" for i in range(len(concepts))]
        sims = [c["similarity"] for c in concepts]
        colors = ["#31a354" if c["covered"] else "#de2d26" for c in concepts]
        fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(concepts)), 4.5))
        ax.bar(labels, sims, color=colors)
        ax.axhline(saf_ex["tau"], ls="--", color="#333", label=f"tau={saf_ex['tau']}")
        ax.set_ylabel("concept similarity"); ax.set_ylim(0, 1)
        ax.set_title(f"Phase 2F — rubric-concept coverage (SAF example, score={ex['score']}, "
                     f"{ex['n_covered']}/{ex['n_concepts']} covered)")
        ax.legend()
        fig.tight_layout(); fig.savefig(fig_dir / "phase2f_concept_example.png", dpi=120); plt.close(fig)
    log.info(f"wrote Phase 2F figures to {fig_dir}")


if __name__ == "__main__":
    import sys

    run_xai(only=sys.argv[1:] or None)
