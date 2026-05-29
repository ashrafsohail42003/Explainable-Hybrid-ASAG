"""Two-view text preprocessing pipeline.

* **Encoder view** (for SBERT / DeBERTa later): NFKC + whitespace normalization
  only. Casing, punctuation, stopwords are preserved.
* **Feature view** (for handcrafted features): spaCy lemmatization + lowercase
  + punctuation removal + stopword removal, with negators preserved.

Writes one parquet per (dataset, view) under ``data/processed/<dataset>/``,
plus a small JSON sidecar with row counts and a sample row, for traceability.

Usage::

    python -m asag.data.preprocess
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from asag.config import DataConfig, ensure_dirs, load_data_config
from asag.data.loaders import load_all
from asag.data.splits import make_stratified_kfold
from asag.utils.logging import get_logger
from asag.utils.seed import set_global_seed

log = get_logger()

_WS_RE = re.compile(r"\s+")


def encoder_view(text: str, *, normalize_unicode: str = "NFKC", collapse_whitespace: bool = True) -> str:
    """Minimal normalization — preserves all transformer signal."""
    if not isinstance(text, str):
        return ""
    out = unicodedata.normalize(normalize_unicode, text)
    if collapse_whitespace:
        out = _WS_RE.sub(" ", out).strip()
    return out


@dataclass
class FeatureViewOptions:
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    preserve_negators: tuple[str, ...] = ()


def _build_negator_set(words: Iterable[str]) -> set[str]:
    return {w.lower().strip() for w in words if w and w.strip()}


def feature_view_batch(texts: list[str], nlp, opts: FeatureViewOptions) -> list[str]:
    """Process a list of texts through spaCy and return processed strings.

    Tokens that are alphabetic (or are negators) survive; punctuation is
    dropped; stopwords are removed unless they appear in ``preserve_negators``.
    """
    negators = _build_negator_set(opts.preserve_negators)
    out: list[str] = []
    # spaCy nlp.pipe handles batching efficiently
    docs = nlp.pipe([t if isinstance(t, str) else "" for t in texts], batch_size=64)
    for doc in docs:
        kept: list[str] = []
        for tok in doc:
            if tok.is_space:
                continue
            t_low = tok.text.lower()
            is_negator = t_low in negators or tok.lemma_.lower() in negators
            if opts.remove_punctuation and tok.is_punct and not is_negator:
                continue
            if opts.remove_stopwords and tok.is_stop and not is_negator:
                continue
            lemma = tok.lemma_ if tok.lemma_ else tok.text
            if opts.lowercase:
                lemma = lemma.lower()
            kept.append(lemma)
        out.append(" ".join(kept).strip())
    return out


def _load_spacy(model: str):
    import spacy
    try:
        return spacy.load(model, disable=["ner", "parser"])
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{model}' not installed. Run: python -m spacy download {model}"
        ) from e


def _process_dataset(name: str, df: pd.DataFrame, cfg: DataConfig, nlp) -> dict:
    """Write encoder and feature views to data/processed/<name>/."""
    enc_cfg = cfg.preprocessing.encoder_view
    feat_cfg = cfg.preprocessing.feature_view

    df = df.copy()
    # encoder view: per-column normalized text
    for col in ("question", "reference_answer", "student_answer"):
        df[f"{col}_enc"] = df[col].map(
            lambda s: encoder_view(
                s,
                normalize_unicode=enc_cfg.normalize_unicode,
                collapse_whitespace=enc_cfg.collapse_whitespace,
            )
        )

    # feature view: only student_answer + reference_answer (questions cheap to reuse)
    opts = FeatureViewOptions(
        lowercase=feat_cfg.lowercase,
        remove_punctuation=feat_cfg.remove_punctuation,
        remove_stopwords=feat_cfg.remove_stopwords,
        preserve_negators=tuple(feat_cfg.preserve_negators),
    )
    for col in ("student_answer", "reference_answer", "question"):
        df[f"{col}_feat"] = feature_view_batch(df[col].tolist(), nlp, opts)

    # stratified k-fold for datasets without official splits
    if (df["split"].unique() == ["all"]).all():
        df["fold"] = make_stratified_kfold(
            df, k=cfg.splits.cv_k_folds, seed=cfg.seed, stratify_on=cfg.splits.stratify_on,
        ).values
    else:
        df["fold"] = -1

    out_dir = cfg.paths.processed / name
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_cols = ["question_id", "question_enc", "reference_answer_enc", "student_answer_enc",
                "score", "label", "dataset", "domain", "split", "fold"]
    feat_cols = ["question_id", "question_feat", "reference_answer_feat", "student_answer_feat",
                 "score", "label", "dataset", "domain", "split", "fold"]

    enc_path = out_dir / "encoder.parquet"
    feat_path = out_dir / "feature.parquet"
    df[enc_cols].to_parquet(enc_path, index=False)
    df[feat_cols].to_parquet(feat_path, index=False)

    # jsonl backups
    df[enc_cols].to_json(out_dir / "encoder.jsonl", orient="records", lines=True, force_ascii=False)
    df[feat_cols].to_json(out_dir / "feature.jsonl", orient="records", lines=True, force_ascii=False)

    sidecar = {
        "dataset": name,
        "n_rows": int(len(df)),
        "splits": {sp: int((df["split"] == sp).sum()) for sp in sorted(df["split"].unique())},
        "encoder_view_file": str(enc_path),
        "feature_view_file": str(feat_path),
        "sample": {
            "encoder": df[enc_cols].iloc[0].to_dict() if len(df) else {},
            "feature": df[feat_cols].iloc[0].to_dict() if len(df) else {},
        },
    }
    (out_dir / "_sidecar.json").write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    log.info(f"{name}: wrote encoder + feature parquets ({len(df)} rows)")
    return sidecar


def run_all(cfg: DataConfig) -> dict[str, dict]:
    ensure_dirs(cfg)
    set_global_seed(cfg.seed)
    nlp = _load_spacy(cfg.preprocessing.feature_view.spacy_model)
    frames = load_all(cfg)
    results: dict[str, dict] = {}
    for name, df in frames.items():
        results[name] = _process_dataset(name, df, cfg, nlp)
    return results


if __name__ == "__main__":
    cfg = load_data_config()
    run_all(cfg)
