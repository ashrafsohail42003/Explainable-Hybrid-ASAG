"""Phase 2B feature-validation report.

Sanity-checks the materialized feature matrices: per-feature coverage (non-NaN
fraction) per dataset, and a target-association score (|Spearman| between each
feature and the grading target). Writes JSON + CSV under reports/phase2b/ and
two heatmap figures under reports/figures/.

Target = numeric ``score`` when it varies, else label-encoded ``label`` (rank
correlation; for nominal labels like SemEval's 5-way this is indicative only).

Usage::

    python -m asag.features.validate_features
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from asag.config import DataConfig, load_data_config
from asag.utils.logging import get_logger

log = get_logger()


def _target(df: pd.DataFrame) -> tuple[pd.Series, str]:
    score = pd.to_numeric(df["score"], errors="coerce")
    if score.notna().any() and score.nunique(dropna=True) > 1:
        return score, "score"
    codes = pd.Categorical(df["label"].astype(str)).codes
    return pd.Series(codes, index=df.index, dtype="float64"), "label_code"


def _associations(df: pd.DataFrame, feature_cols: list[str]) -> tuple[dict, str]:
    target, kind = _target(df)
    out: dict[str, float] = {}
    for c in feature_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        if col.notna().sum() < 2 or col.nunique(dropna=True) < 2:
            out[c] = float("nan")
            continue
        out[c] = abs(float(col.corr(target, method="spearman")))
    return out, kind


def _heatmap(matrix: pd.DataFrame, title: str, cbar: str, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, 0.32 * matrix.shape[1]), max(3, 0.5 * matrix.shape[0])))
    im = ax.imshow(matrix.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar, fraction=0.025, pad=0.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    log.info(f"wrote figure {out_path}")


def run_all(cfg: DataConfig | None = None) -> dict:
    cfg = cfg or load_data_config()
    coverage_rows: dict[str, dict] = {}
    assoc_rows: dict[str, dict] = {}
    report: dict[str, dict] = {}

    for name in sorted(cfg.datasets):
        fpath = cfg.paths.processed / name / "features.parquet"
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        keys = {"question_id", "score", "label", "dataset", "domain", "split", "fold"}
        feature_cols = [c for c in df.columns if c not in keys]

        coverage = {c: round(float(df[c].notna().mean()), 4) for c in feature_cols}
        assoc, kind = _associations(df, feature_cols)
        coverage_rows[name] = coverage
        assoc_rows[name] = assoc
        report[name] = {
            "n_rows": int(len(df)),
            "n_features": len(feature_cols),
            "target_kind": kind,
            "fully_nan_features": [c for c in feature_cols if coverage[c] == 0.0],
            "coverage": coverage,
            "target_association_abs_spearman": assoc,
        }
        log.info(f"{name}: {len(feature_cols)} features, target={kind}, "
                 f"{len(report[name]['fully_nan_features'])} fully-NaN")

    if not report:
        raise RuntimeError("No features.parquet found. Run `make features` first.")

    out_dir = cfg.paths.reports / "phase2b"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    cov_df = pd.DataFrame(coverage_rows).T.sort_index()
    cov_df.to_csv(out_dir / "feature_coverage.csv")
    assoc_df = pd.DataFrame(assoc_rows).T.reindex(index=cov_df.index, columns=cov_df.columns)

    _heatmap(cov_df.fillna(0.0), "Phase 2B — feature coverage (non-NaN fraction)",
             "coverage", cfg.paths.figures / "phase2b_coverage_heatmap.png")
    _heatmap(assoc_df.fillna(0.0), "Phase 2B — |Spearman| feature vs. target",
             "|rho|", cfg.paths.figures / "phase2b_target_assoc.png")

    log.info(f"wrote {out_dir / 'feature_validation.json'} and {out_dir / 'feature_coverage.csv'}")
    return report


if __name__ == "__main__":
    run_all()
