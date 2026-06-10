"""Phase 3 — ablation studies (the experimental heart of the paper).

Proves each fusion branch earns its keep. We reuse the *unchanged* Phase 2C
evaluation protocol (official_split / kfold / per_prompt, multi-seed) but restrict
``Bundle.feature_cols`` to a feature subset, so the only thing that changes between
variants is which branch's features the head can see — the ΔQWK/Δmetric is then
attributable to the branch, not to a protocol or HPO difference.

Branches, by feature prefix (the report's A/B/C, plus D when the hybrid is on):

* **A — semantic**  : ``sem_*`` (SBERT cosine + interaction summaries)
* **B — linguistic**: ``lex_ / len_ / tfidf_ / neg_ / ner_``
* **C — rubric**    : ``rub_*`` (concept coverage)
* **D — neural**    : ``neural_*`` (DeBERTa cross-encoder out-of-fold signals; present
  only once ``neural_oof.parquet`` has been produced on Colab — otherwise the ``-D`` /
  ``only-D`` variants report 0 features and are skipped). This is the ablation that
  answers "how much does the transformer add on top of the interpretable features?".

Variants: ``full``, ``-A/-B/-C/-D`` (drop one), ``only-A/.../only-D`` (keep one),
and ``-neg`` (drop the negation-cue features — the report's preprocessing ablation).

A single lightly-regularized head config (``subsample = colsample = 0.8``) is used
for *every* variant so the comparison is fair and the seed std is honest (non-zero).
The encoder/head ablations (BERT/RoBERTa/DeBERTa backbone, CORAL/CORN) still need the
torch slice; branch D captures the cross-encoder's *fused* contribution here.

    python -m asag.models.ablations [<name>...]      # no args = all datasets with features
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from asag.config import DataConfig, LightGBMCfg, ensure_dirs, load_data_config
from asag.models.data import Bundle, load_bundle
from asag.models.evaluate import _eval_kfold, _eval_official
from asag.models.fusion import LIGHTGBM_AVAILABLE
from asag.models.tasks import REGISTRY, get_spec
from asag.utils.logging import get_logger
from asag.utils.seed import set_global_seed

log = get_logger()

PHASE3_SCHEMA_VERSION = "3.1"

_B_PREFIXES = ("lex_", "len_", "tfidf_", "neg_", "ner_")
# negation-cue features (preprocessing ablation): the neg_* family + the
# negation-scope lexical overlap. Dropping these = "without negation cues".
_NEG_COLS = ("lex_content_word_overlap_neg",)

# variant -> human label for reports/figures
VARIANTS = ("full", "-A", "-B", "-C", "-D",
            "only-A", "only-B", "only-C", "only-D", "-neg")


def branch_of(col: str) -> str:
    if col.startswith("sem_"):
        return "A"
    if col.startswith("rub_"):
        return "C"
    if col.startswith("neural_"):
        return "D"
    return "B"


def _groups(feature_cols: list[str]) -> dict[str, list[str]]:
    A = [c for c in feature_cols if c.startswith("sem_")]
    C = [c for c in feature_cols if c.startswith("rub_")]
    D = [c for c in feature_cols if c.startswith("neural_")]
    B = [c for c in feature_cols if c.startswith(_B_PREFIXES)]
    return {"A": A, "B": B, "C": C, "D": D}


def _neg_cols(feature_cols: list[str]) -> list[str]:
    return [c for c in feature_cols if c.startswith("neg_") or c in _NEG_COLS]


def variant_cols(feature_cols: list[str], variant: str) -> list[str]:
    g = _groups(feature_cols)
    if variant == "full":
        return list(feature_cols)
    if variant == "-A":
        return [c for c in feature_cols if c not in set(g["A"])]
    if variant == "-B":
        return [c for c in feature_cols if c not in set(g["B"])]
    if variant == "-C":
        return [c for c in feature_cols if c not in set(g["C"])]
    if variant == "-D":
        return [c for c in feature_cols if c not in set(g["D"])]
    if variant == "only-A":
        return list(g["A"])
    if variant == "only-B":
        return list(g["B"])
    if variant == "only-C":
        return list(g["C"])
    if variant == "only-D":
        return list(g["D"])
    if variant == "-neg":
        drop = set(_neg_cols(feature_cols))
        return [c for c in feature_cols if c not in drop]
    raise ValueError(f"unknown variant {variant!r}")


def _ablation_head(cfg: DataConfig) -> LightGBMCfg:
    """One regularized head config used for every variant (fair + honest std)."""
    return LightGBMCfg(**{**cfg.model.lightgbm.model_dump(),
                          "subsample": 0.8, "colsample_bytree": 0.8})


def _headline(evals: dict, spec) -> dict:
    split = spec.test_splits[-1] if spec.protocol == "official_split" else "cv"
    return evals.get(split, {}).get("gbm", {}).get(spec.headline, {})


def ablate_dataset(name: str, cfg: DataConfig, head: LightGBMCfg) -> dict | None:
    spec = get_spec(name)
    bundle = load_bundle(name, cfg, spec)
    if bundle is None:
        log.warning(f"{name}: features.parquet missing — skipping")
        return None

    out: dict[str, dict] = {}
    for v in VARIANTS:
        cols = variant_cols(bundle.feature_cols, v)
        if not cols:
            out[v] = {"n_features": 0, "skipped": "no features in this branch"}
            continue
        sub = replace(bundle, feature_cols=cols)
        evals = (_eval_official(sub, cfg, head) if spec.protocol == "official_split"
                 else _eval_kfold(sub, cfg, head))
        out[v] = {"n_features": len(cols), "headline": _headline(evals, spec)}

    full = out["full"]["headline"].get("mean", float("nan"))
    for v in VARIANTS:
        m = out[v].get("headline", {}).get("mean")
        out[v]["delta_vs_full"] = (None if m is None or not np.isfinite(full)
                                   else round(float(m - full), 4))
    log.info(f"{name}: {spec.headline} full={full:.4f} "
             f"| -A Δ={out['-A'].get('delta_vs_full')} "
             f"-B Δ={out['-B'].get('delta_vs_full')} "
             f"-C Δ={out['-C'].get('delta_vs_full')} "
             f"-neg Δ={out['-neg'].get('delta_vs_full')}")
    return {"task_type": spec.task_type, "metric": spec.headline,
            "headline_split": spec.test_splits[-1] if spec.protocol == "official_split" else "cv",
            "variants": out}


def run_ablations(cfg: DataConfig | None = None, only: list[str] | None = None) -> dict:
    cfg = cfg or load_data_config()
    ensure_dirs(cfg)
    set_global_seed(cfg.seed)
    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError("lightgbm is not installed; run `uv pip install lightgbm`")

    head = _ablation_head(cfg)
    names = only or [n for n in REGISTRY if (cfg.paths.processed / n / "features.parquet").exists()]
    results = {n: r for n in names if (r := ablate_dataset(n, cfg, head)) is not None}
    if results:
        _write_reports(cfg, results, head)
        _write_figures(cfg, results)
    return results


# ----------------------------- reporting ---------------------------------

def _write_reports(cfg: DataConfig, results: dict, head: LightGBMCfg) -> None:
    out_dir = cfg.paths.reports / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablations.json").write_text(json.dumps(
        {"schema_version": PHASE3_SCHEMA_VERSION,
         "seeds": list(cfg.model.seeds),
         "head": head.model_dump(),
         "datasets": results}, indent=2, default=str), encoding="utf-8")

    rows = []
    for name, r in results.items():
        for v, d in r["variants"].items():
            h = d.get("headline", {})
            rows.append({"dataset": name, "metric": r["metric"], "variant": v,
                         "n_features": d.get("n_features"),
                         "mean": round(h.get("mean", float("nan")), 4) if h else None,
                         "std": round(h.get("std", float("nan")), 4) if h else None,
                         "delta_vs_full": d.get("delta_vs_full")})
    pd.DataFrame(rows).to_csv(out_dir / "ablations.csv", index=False)
    log.info(f"wrote {out_dir}/ ablations.json, ablations.csv")


def _write_figures(cfg: DataConfig, results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = cfg.paths.figures
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())

    # 1) headline per dataset for full / -A / -B / -C (with seed-std error bars).
    show = ["full", "-A", "-B", "-C"]
    colors = {"full": "#2c7fb8", "-A": "#fb6a4a", "-B": "#fdae6b", "-C": "#74c476"}
    x = np.arange(len(names)); w = 0.2
    fig, ax = plt.subplots(figsize=(max(8, 1.9 * len(names)), 5))
    for k, v in enumerate(show):
        means = [results[n]["variants"][v].get("headline", {}).get("mean", np.nan) for n in names]
        stds = [results[n]["variants"][v].get("headline", {}).get("std", 0.0) or 0.0 for n in names]
        ax.bar(x + (k - 1.5) * w, means, w, yerr=stds, capsize=3, label=v, color=colors[v])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({results[n]['metric']})" for n in names], fontsize=8)
    ax.set_ylabel("headline metric"); ax.set_title("Phase 3 — branch ablations (drop one branch)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "phase3_branch_ablation.png", dpi=120); plt.close(fig)

    # 2) Δ (drop − full) per branch + negation: how much removing each branch hurts.
    # Include -D (neural) only when the hybrid is actually present (some dataset has
    # neural features), else a flat 0 bar would falsely read as "neural adds nothing".
    has_neural = any(results[n]["variants"].get("-D", {}).get("n_features", 0)
                     for n in names)
    drops = ["-A", "-B", "-C"] + (["-D"] if has_neural else []) + ["-neg"]
    w2 = 0.8 / len(drops)
    fig, ax = plt.subplots(figsize=(max(8, 1.9 * len(names)), 5))
    for k, v in enumerate(drops):
        deltas = [results[n]["variants"][v].get("delta_vs_full") or 0.0 for n in names]
        ax.bar(x + (k - (len(drops) - 1) / 2) * w2, deltas, w2, label=v)
    ax.axhline(0.0, color="#333", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Δ headline (variant − full)")
    ax.set_title("Phase 3 — contribution of each branch (negative = removing it hurts)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "phase3_branch_delta.png", dpi=120); plt.close(fig)
    log.info(f"wrote Phase 3 figures to {fig_dir}")


if __name__ == "__main__":
    import sys

    run_ablations(only=sys.argv[1:] or None)
