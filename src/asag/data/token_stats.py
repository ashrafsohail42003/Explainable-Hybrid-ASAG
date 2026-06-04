"""Phase 2A — subword token-length study.

The Phase 2A spec says: *use the model-native subword tokenizer and do NOT
pre-tokenize; set ``max_len`` ≈ 128 with dynamic padding.* This module supplies
the empirical justification for that ``max_len`` without materializing any
``input_ids`` — it measures, per dataset, how long the encoder-view text becomes
under two real tokenizers and how much would be truncated at 128 / 256.

Two tokenizers (mirroring the Phase 2C model choices):

* ``bert-base-uncased``      — WordPiece (BERT / RoBERTa-ish family)
* ``microsoft/deberta-v3-base`` — SentencePiece (the chosen cross-encoder)

Two regimes:

* **bi-encoder (SBERT)** — each field alone (``student_answer_enc``,
  ``reference_answer_enc``): the lengths a sentence encoder sees.
* **cross-encoder (DeBERTa)** — the *pair* the model actually consumes, scored
  with ``tokenizer(text_a, text_b)`` so ``[CLS]/[SEP]`` overhead is counted:
  ``reference | student`` and the question-inclusive ``question+reference | student``.

Artifacts (Phase 1 conventions — JSON/CSV under ``reports/``, PNG under
``reports/figures/``):

* ``reports/phase2a/token_lengths.json`` — full percentile tables + recommendations
* ``reports/phase2a/token_lengths.csv``  — flat table for the paper
* ``reports/figures/token_length_bi.png`` / ``token_length_cross.png``

Usage::

    python -m asag.data.token_stats
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from asag.config import DataConfig, load_data_config
from asag.utils.logging import get_logger

log = get_logger()

# (display key, HF id). Loaded lazily; a missing/offline tokenizer is skipped.
TOKENIZERS: list[tuple[str, str]] = [
    ("bert_wordpiece", "bert-base-uncased"),
    ("deberta_v3_sp", "microsoft/deberta-v3-base"),
]

# max_len candidates we report truncation against.
TRUNC_AT = (128, 256)

PERCENTILES = (50, 90, 95, 99)


def _percentile_block(lengths: np.ndarray) -> dict:
    """Summary stats + truncation rates for one length array."""
    if lengths.size == 0:
        return {}
    block = {
        "n": int(lengths.size),
        "mean": round(float(lengths.mean()), 2),
        "max": int(lengths.max()),
    }
    for p in PERCENTILES:
        block[f"p{p}"] = int(np.percentile(lengths, p))
    for t in TRUNC_AT:
        block[f"pct_over_{t}"] = round(float((lengths > t).mean() * 100.0), 2)
    return block


def _lengths_single(tok, texts: list[str]) -> np.ndarray:
    """Token counts (with special tokens) for a list of single texts."""
    enc = tok(texts, add_special_tokens=True)["input_ids"]
    return np.fromiter((len(ids) for ids in enc), dtype=np.int32, count=len(enc))


def _lengths_pair(tok, texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    """Token counts for (a, b) pairs — the cross-encoder input."""
    enc = tok(texts_a, texts_b, add_special_tokens=True)["input_ids"]
    return np.fromiter((len(ids) for ids in enc), dtype=np.int32, count=len(enc))


def _load_tokenizers() -> list[tuple[str, object]]:
    from transformers import AutoTokenizer

    loaded: list[tuple[str, object]] = []
    for key, hf_id in TOKENIZERS:
        try:
            loaded.append((key, AutoTokenizer.from_pretrained(hf_id)))
            log.info(f"loaded tokenizer {key} ({hf_id})")
        except Exception as e:  # offline / missing sentencepiece, etc.
            log.warning(f"skipping tokenizer {key} ({hf_id}): {e}")
    if not loaded:
        raise RuntimeError(
            "No tokenizers could be loaded. Check network access and that "
            "`transformers` + `sentencepiece` are installed."
        )
    return loaded


def _str_col(df: pd.DataFrame, col: str) -> list[str]:
    return df[col].fillna("").astype(str).tolist()


def study_dataset(name: str, df: pd.DataFrame, tokenizers) -> list[dict]:
    """Return flat rows: one per (dataset, tokenizer, regime, field)."""
    student = _str_col(df, "student_answer_enc")
    reference = _str_col(df, "reference_answer_enc")
    question = _str_col(df, "question_enc")
    q_ref = [f"{q} {r}".strip() for q, r in zip(question, reference)]

    rows: list[dict] = []
    for tkey, tok in tokenizers:
        # bi-encoder: single fields
        for field, texts in (("student", student), ("reference", reference)):
            block = _percentile_block(_lengths_single(tok, texts))
            rows.append({"dataset": name, "tokenizer": tkey, "regime": "bi",
                         "field": field, **block})
        # cross-encoder: pairs
        for field, a, b in (
            ("ref|student", reference, student),
            ("q+ref|student", q_ref, student),
        ):
            block = _percentile_block(_lengths_pair(tok, a, b))
            rows.append({"dataset": name, "tokenizer": tkey, "regime": "cross",
                         "field": field, **block})
    return rows


# Report target (the spec's "short answers -> max_len ~ 128"). A dataset is
# flagged as needing an override when truncating at this target loses more than
# OUTLIER_TRUNC_PCT of its rows.
TARGET_MAX_LEN = 128
OUTLIER_TRUNC_PCT = 5.0


def _ladder(p99: int) -> int:
    return 128 if p99 <= 128 else (256 if p99 <= 256 else 384)


def _recommend(flat: pd.DataFrame) -> dict:
    """Per-regime recommendation, field- and dataset-aware.

    A single global max_len is misleading here: most datasets' answers fit 128,
    but one dataset (SAF) has very long multi-point *reference* solutions that
    would force a global value up. So we anchor the primary recommendation at
    the report target (128) and break out the datasets whose binding field
    exceeds it as explicit per-dataset overrides.
    """
    rec: dict = {}
    for regime in ("bi", "cross"):
        sub = flat[flat["regime"] == regime]
        if sub.empty:
            continue
        per_dataset: dict = {}
        for ds, g in sub.groupby("dataset"):
            worst = g.loc[g["p99"].idxmax()]  # the field that binds max_len
            per_dataset[ds] = {
                "binding_field": worst["field"],
                "tokenizer": worst["tokenizer"],
                "p95": int(worst["p95"]),
                "p99": int(worst["p99"]),
                "max": int(worst["max"]),
                "pct_over_128": float(worst["pct_over_128"]),
                "pct_over_256": float(worst["pct_over_256"]),
                "suggested_max_len": _ladder(int(worst["p99"])),
            }
        outliers = sorted(
            ds for ds, v in per_dataset.items()
            if v["pct_over_128"] > OUTLIER_TRUNC_PCT
        )
        fits = {ds: v for ds, v in per_dataset.items() if ds not in outliers}
        max_trunc_within_fits = max((v["pct_over_128"] for v in fits.values()), default=0.0)
        rec[regime] = {
            "primary_max_len": TARGET_MAX_LEN,
            "note": (
                f"{TARGET_MAX_LEN} covers all datasets except {outliers or 'none'} "
                f"(<= {round(max_trunc_within_fits, 2)}% truncation on the rest)"
            ),
            "outliers_needing_override": outliers,
            "per_dataset": per_dataset,
        }
    rec["padding"] = "dynamic per-batch (pad to longest in batch, not to max_len)"
    return rec


def _plot(flat: pd.DataFrame, regime: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = flat[flat["regime"] == regime].copy()
    if sub.empty:
        return
    sub["group"] = sub["dataset"] + "\n" + sub["field"]
    pivot = sub.pivot_table(index="group", columns="tokenizer", values="p95")
    ax = pivot.plot(kind="bar", figsize=(max(8, 1.1 * len(pivot)), 5))
    for t in TRUNC_AT:
        ax.axhline(t, ls="--", lw=1, color="grey")
        ax.text(ax.get_xlim()[1], t, f" {t}", va="center", fontsize=8, color="grey")
    ax.set_ylabel("p95 subword tokens")
    ax.set_title(f"Phase 2A — {regime}-encoder token length (p95) by dataset/field")
    ax.legend(title="tokenizer", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    log.info(f"wrote figure {out_path}")


def run_all(cfg: DataConfig) -> dict:
    tokenizers = _load_tokenizers()
    processed = cfg.paths.processed

    all_rows: list[dict] = []
    for name in sorted(cfg.datasets):
        if not cfg.datasets[name].enabled:
            continue
        enc_path = processed / name / "encoder.parquet"
        if not enc_path.exists():
            log.warning(f"{name}: no encoder.parquet — run `make preprocess` first; skipping")
            continue
        df = pd.read_parquet(enc_path)
        log.info(f"{name}: measuring {len(df)} rows")
        all_rows.extend(study_dataset(name, df, tokenizers))

    if not all_rows:
        raise RuntimeError("No processed datasets found. Run `make preprocess` first.")

    flat = pd.DataFrame(all_rows)
    recommendations = _recommend(flat)

    out_dir = cfg.paths.reports / "phase2a"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "token_lengths.csv"
    json_path = out_dir / "token_lengths.json"
    flat.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "tokenizers": {k: v for k, v in TOKENIZERS},
                "regimes": {
                    "bi": "single field (SBERT bi-encoder)",
                    "cross": "tokenizer(text_a, text_b) pair (DeBERTa cross-encoder)",
                },
                "truncation_thresholds": list(TRUNC_AT),
                "recommendations": recommendations,
                "rows": all_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(f"wrote {json_path} and {csv_path}")

    _plot(flat, "bi", cfg.paths.figures / "token_length_bi.png")
    _plot(flat, "cross", cfg.paths.figures / "token_length_cross.png")

    log.info(f"recommendations: {json.dumps(recommendations)}")
    return {"recommendations": recommendations, "n_rows": len(all_rows)}


if __name__ == "__main__":
    run_all(load_data_config())
