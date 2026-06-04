"""Two-view text preprocessing pipeline.

* **Encoder view** (for SBERT / DeBERTa later): NFKC + whitespace normalization
  only. Casing, punctuation, stopwords are preserved.
* **Feature view** (for handcrafted features): spaCy lemmatization + lowercase
  + punctuation removal + stopword removal, with negators preserved. A parallel
  **negation-scope** variant (``*_feat_neg`` columns) additionally prefixes the
  tokens within a negator's scope with ``neg_`` (e.g. "does not forward" ->
  "neg_forward"), stopping at a clause boundary. ``*_feat`` stays plain so the
  marked/plain pair is an ablation knob for Phase 2B.

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
    negation_scope: bool = False
    negation_window: int = 4


def _build_negator_set(words: Iterable[str]) -> set[str]:
    return {w.lower().strip() for w in words if w and w.strip()}


def feature_view_batch(
    texts: list[str], nlp, opts: FeatureViewOptions
) -> tuple[list[str], list[str]]:
    """Process a list of texts through spaCy.

    Tokens that are alphabetic (or are negators) survive; punctuation is
    dropped; stopwords are removed unless they appear in ``preserve_negators``.

    Returns ``(plain, marked)`` parallel lists. ``plain`` is the lemmatized
    feature view; ``marked`` additionally prefixes tokens inside a negator's
    scope with ``neg_``. A negator opens a scope of ``opts.negation_window``
    subsequent kept tokens, closed early when a clause boundary (punctuation or
    a coordinating conjunction) is observed. When ``opts.negation_scope`` is
    False, ``marked`` is identical to ``plain``.
    """
    negators = _build_negator_set(opts.preserve_negators)
    plain: list[str] = []
    marked: list[str] = []
    # spaCy nlp.pipe handles batching efficiently
    docs = nlp.pipe([t if isinstance(t, str) else "" for t in texts], batch_size=64)
    for doc in docs:
        kept: list[str] = []
        kept_marked: list[str] = []
        scope_left = 0  # tokens still within an open negation scope
        for tok in doc:
            if tok.is_space:
                continue
            t_low = tok.text.lower()
            is_negator = t_low in negators or tok.lemma_.lower() in negators
            # Clause boundary closes any open scope (before the token is dropped).
            if opts.negation_scope and not is_negator and (
                tok.is_punct or tok.pos_ == "CCONJ"
            ):
                scope_left = 0
            if opts.remove_punctuation and tok.is_punct and not is_negator:
                continue
            if opts.remove_stopwords and tok.is_stop and not is_negator:
                continue
            lemma = tok.lemma_ if tok.lemma_ else tok.text
            if opts.lowercase:
                lemma = lemma.lower()
            kept.append(lemma)

            if opts.negation_scope:
                if is_negator:
                    # The negator itself is a cue; mark it and (re)open the scope.
                    kept_marked.append(f"neg_{lemma}")
                    scope_left = opts.negation_window
                elif scope_left > 0:
                    kept_marked.append(f"neg_{lemma}")
                    scope_left -= 1
                else:
                    kept_marked.append(lemma)

        plain.append(" ".join(kept).strip())
        marked.append(" ".join(kept_marked).strip() if opts.negation_scope else " ".join(kept).strip())
    return plain, marked


def _load_spacy(model: str):
    import spacy
    try:
        return spacy.load(model, disable=["ner", "parser"])
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{model}' not installed. Run: python -m spacy download {model}"
        ) from e


def dedupe_within_question(df: pd.DataFrame, score_col: str = "score") -> tuple[pd.DataFrame, int]:
    """Drop exact (question_id, student_answer) duplicates within a dataset.

    When duplicates differ on score, we keep the row whose score is closest
    to the group's median (median of group scores). This avoids "best/worst
    answer" bias from naively keeping `first`/`last`. Returns
    (deduped_df, n_dropped).
    """
    if "question_id" not in df.columns or "student_answer" not in df.columns:
        return df, 0
    n_before = len(df)
    work = df.copy()
    # group median (broadcast back to row level)
    grp_med = work.groupby(
        ["question_id", "student_answer"], dropna=False, sort=False
    )[score_col].transform("median")
    work["_dist"] = (work[score_col] - grp_med).abs()
    # Stable sort: closest to median first; ties broken by original order
    work = work.sort_values(["question_id", "student_answer", "_dist"], kind="mergesort")
    deduped = (
        work.drop_duplicates(subset=["question_id", "student_answer"], keep="first")
            .drop(columns=["_dist"])
            .reset_index(drop=True)
    )
    return deduped, n_before - len(deduped)


def _process_dataset(name: str, df: pd.DataFrame, cfg: DataConfig, nlp) -> dict:
    """Write encoder and feature views to data/processed/<name>/."""
    enc_cfg = cfg.preprocessing.encoder_view
    feat_cfg = cfg.preprocessing.feature_view

    # Dedup datasets without official splits that have known exact-duplicate
    # issues. Keep the median-score row per dup group.
    n_dropped = 0
    if name in {"mohler", "powergrading", "mindreading"}:
        df, n_dropped = dedupe_within_question(df)
        if n_dropped:
            log.info(f"{name}: dropped {n_dropped} exact duplicate (question, answer) rows")

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

    # feature view (+ parallel negation-scope variant)
    opts = FeatureViewOptions(
        lowercase=feat_cfg.lowercase,
        remove_punctuation=feat_cfg.remove_punctuation,
        remove_stopwords=feat_cfg.remove_stopwords,
        preserve_negators=tuple(feat_cfg.preserve_negators),
        negation_scope=feat_cfg.negation_scope.enabled,
        negation_window=feat_cfg.negation_scope.window,
    )
    for col in ("student_answer", "reference_answer", "question"):
        plain, marked = feature_view_batch(df[col].tolist(), nlp, opts)
        df[f"{col}_feat"] = plain
        df[f"{col}_feat_neg"] = marked

    # stratified k-fold for datasets without official splits
    unique_splits = set(df["split"].astype(str).unique())
    if unique_splits == {"all"}:
        df["fold"] = make_stratified_kfold(
            df, k=cfg.splits.cv_k_folds, seed=cfg.seed, stratify_on=cfg.splits.stratify_on,
        ).values
    else:
        df["fold"] = -1

    out_dir = cfg.paths.processed / name
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_cols = ["question_id", "question_enc", "reference_answer_enc", "student_answer_enc",
                "score", "label", "dataset", "domain", "split", "fold"]
    feat_cols = ["question_id",
                 "question_feat", "reference_answer_feat", "student_answer_feat",
                 "question_feat_neg", "reference_answer_feat_neg", "student_answer_feat_neg",
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
        "n_dropped_dedup": int(n_dropped),
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
