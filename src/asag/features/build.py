"""Phase 2B feature-build orchestrator.

Reads the two Phase 2A views per dataset, runs the enabled branches, and writes
one interpretable ``features.parquet`` per dataset (+ JSONL sample, sidecar) and
a global ``feature_dictionary.json``. The SBERT model and spaCy(NER+sentencizer)
pipe are loaded once and shared across datasets and branches.

Usage::

    python -m asag.features.build
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from asag.config import DataConfig, ensure_dirs, load_data_config
from asag.features import FEATURES_SCHEMA_VERSION, entities, lexical, negation, rubric, semantic, tfidf
from asag.features.text_utils import load_feature_nlp
from asag.utils.logging import get_logger
from asag.utils.seed import set_global_seed

log = get_logger()

KEY_COLUMNS = ["question_id", "score", "label", "dataset", "domain", "split", "fold"]
ENC_COLS = ["question_enc", "reference_answer_enc", "student_answer_enc"]
FEAT_COLS = ["question_feat", "reference_answer_feat", "student_answer_feat",
             "question_feat_neg", "reference_answer_feat_neg", "student_answer_feat_neg"]

# Per-branch metadata for the feature dictionary.
BRANCH_META = {
    "lexical": {"view": "feature", "columns": lexical.COLUMNS, "ref_dep": set(lexical.REF_DEPENDENT)},
    "tfidf": {"view": "feature", "columns": tfidf.COLUMNS, "ref_dep": set(tfidf.COLUMNS)},
    "negation": {"view": "feature", "columns": negation.COLUMNS, "ref_dep": set(negation.REF_DEPENDENT)},
    "entities": {"view": "encoder", "columns": entities.COLUMNS, "ref_dep": set(entities.REF_DEPENDENT)},
    "semantic": {"view": "encoder", "columns": semantic.SCALAR_COLUMNS, "ref_dep": set(semantic.SCALAR_COLUMNS)},
    "rubric": {"view": "encoder", "columns": rubric.COLUMNS, "ref_dep": set(rubric.COLUMNS)},
}

DESCRIPTIONS = {
    "lex_token_overlap": "Jaccard of unigram content-word sets (student vs reference)",
    "lex_token_overlap_recall": "Fraction of reference content words present in the student answer",
    "lex_token_overlap_prec": "Fraction of student content words present in the reference",
    "lex_token_dice": "Dice coefficient of unigram content-word sets",
    "lex_bigram_overlap": "Jaccard of bigram sets",
    "lex_trigram_overlap": "Jaccard of trigram sets",
    "lex_content_word_overlap_neg": "Unigram Jaccard on the negation-scope view (polarity-aware twin)",
    "len_ratio_sr": "student_tokens / reference_tokens",
    "len_logratio_sr": "log((student_tokens+1)/(reference_tokens+1))",
    "len_student_tokens": "Student content-word token count",
    "len_student_chars": "Student answer character length (raw encoder view)",
    "len_student_uniq_ratio": "Type-token ratio of the student answer",
    "tfidf_cosine": "Cosine of per-dataset TF-IDF vectors (student vs reference)",
    "neg_student_count": "Count of negation-scope-marked tokens in the student answer",
    "neg_student_present": "1 if the student answer contains any negation scope",
    "neg_reference_count": "Count of negation-scope-marked tokens in the reference",
    "neg_polarity_mismatch": "1 if exactly one of {student, reference} carries negation (XOR)",
    "neg_overlap_delta": "Plain unigram overlap minus negation-view overlap",
    "ner_student_count": "Number of named-entity mentions in the student answer",
    "ner_student_uniq_types": "Number of distinct entity labels in the student answer",
    "ner_overlap_count": "Count of entity mentions shared by student and reference",
    "ner_overlap_jaccard": "Jaccard of entity-mention sets",
    "ner_ref_recall": "Fraction of reference entities present in the student answer",
    "sem_cosine": "SBERT cosine(student, reference)",
    "sem_abs_diff_mean": "Mean of |u-v| over SBERT dims (interaction summary)",
    "sem_hadamard_mean": "Mean of u*v over SBERT dims (interaction summary)",
    "rub_n_concepts": "Number of reference sentences treated as rubric concepts",
    "rub_mean_maxsim": "Mean per-concept SBERT similarity to the student answer",
    "rub_min_maxsim": "Weakest per-concept similarity (least-covered concept)",
    "rub_max_maxsim": "Strongest per-concept similarity",
    "rub_coverage_at_tau": "Fraction of concepts with similarity >= tau",
}


def _merge_views(name: str, enc: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """Positional concat of the two row-aligned views, with a hard guard."""
    if len(enc) != len(feat):
        raise ValueError(f"{name}: encoder/feature row count mismatch ({len(enc)} vs {len(feat)})")
    if not (enc["question_id"].astype(str).values == feat["question_id"].astype(str).values).all():
        raise ValueError(f"{name}: encoder/feature question_id sequences differ — views not aligned")
    enc = enc.reset_index(drop=True)
    feat = feat.reset_index(drop=True)
    return pd.concat([enc[KEY_COLUMNS + ENC_COLS], feat[FEAT_COLS]], axis=1)


def build_dataset(name: str, cfg: DataConfig, encoder, nlp) -> dict:
    out_dir = cfg.paths.processed / name
    enc_path, feat_path = out_dir / "encoder.parquet", out_dir / "feature.parquet"
    if not (enc_path.exists() and feat_path.exists()):
        log.warning(f"{name}: missing encoder/feature parquet — run `make preprocess`; skipping")
        return {}

    df = _merge_views(name, pd.read_parquet(enc_path), pd.read_parquet(feat_path))
    has_reference = bool(df["reference_answer_enc"].astype(str).str.strip().ne("").any())
    branches = cfg.features.branches

    parts: list[pd.DataFrame] = [df[KEY_COLUMNS]]
    enabled: list[str] = []
    dense = None

    if branches.lexical:
        parts.append(lexical.compute_lexical(df, cfg)); enabled.append("lexical")
    if branches.tfidf:
        parts.append(tfidf.compute_tfidf(df, cfg)); enabled.append("tfidf")
    if branches.negation:
        parts.append(negation.compute_negation(df, cfg)); enabled.append("negation")
    if branches.entities:
        parts.append(entities.compute_entities(df, cfg, nlp)); enabled.append("entities")
    if branches.semantic:
        scalars, dense = semantic.compute_semantic(df, cfg, encoder)
        parts.append(scalars); enabled.append("semantic")
    if branches.rubric:
        parts.append(rubric.compute_rubric(df, cfg, encoder, nlp)); enabled.append("rubric")

    out = pd.concat(parts, axis=1)
    for part in parts:
        if len(part) != len(df):  # defensive: every branch is row-aligned
            raise ValueError(f"{name}: a branch returned {len(part)} rows, expected {len(df)}")

    out.to_parquet(out_dir / "features.parquet", index=False)
    out.head(50).to_json(out_dir / "features.jsonl", orient="records", lines=True, force_ascii=False)

    feature_cols = [c for c in out.columns if c not in KEY_COLUMNS]
    coverage = {c: round(float(out[c].notna().mean()), 4) for c in feature_cols}
    nan_cols = [c for c in feature_cols if out[c].isna().all()]

    interaction_file = None
    if dense is not None:
        dim = dense.shape[1]
        dcols = [f"sem_int_{i:03d}" for i in range(dim)]
        ddf = pd.concat(
            [df[["question_id"] + KEY_COLUMNS[1:]].reset_index(drop=True),
             pd.DataFrame(dense, columns=dcols)],
            axis=1,
        )
        interaction_file = str(out_dir / "semantic_interaction.parquet")
        ddf.to_parquet(interaction_file, index=False)

    sidecar = {
        "dataset": name,
        "features_schema_version": FEATURES_SCHEMA_VERSION,
        "sbert_model": cfg.features.sbert_model,
        "n_rows": int(len(out)),
        "has_reference": has_reference,
        "branches_enabled": enabled,
        "n_feature_columns": len(feature_cols),
        "feature_columns": feature_cols,
        "nan_columns": nan_cols,
        "coverage": coverage,
        "interaction_vector_file": interaction_file,
        "interaction_dims": int(dense.shape[1]) if dense is not None else 0,
        "sample_row": out.iloc[0].to_dict() if len(out) else {},
    }
    (out_dir / "_features_sidecar.json").write_text(
        json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    log.info(f"{name}: wrote features.parquet ({len(out)} rows, {len(feature_cols)} features, "
             f"has_reference={has_reference})")
    return sidecar


def _write_dictionary(cfg: DataConfig, enabled_branches: list[str]) -> None:
    features: dict[str, dict] = {}
    for branch in enabled_branches:
        meta = BRANCH_META[branch]
        for col in meta["columns"]:
            features[col] = {
                "branch": branch,
                "input_view": meta["view"],
                "dtype": "float64",
                "reference_dependent": col in meta["ref_dep"],
                "nan_when": "no reference answer (asap_sas, mindreading)" if col in meta["ref_dep"] else "never",
                "description": DESCRIPTIONS.get(col, ""),
            }
    doc = {
        "schema_version": FEATURES_SCHEMA_VERSION,
        "sbert_model": cfg.features.sbert_model,
        "branches": {b: {"input_view": BRANCH_META[b]["view"]} for b in enabled_branches},
        "nan_policy": "Reference-dependent features are NaN for rows without a reference "
                      "answer; use NaN-native heads (XGBoost/LightGBM/HistGBM) or impute explicitly.",
        "dense_interaction": {
            "file": "data/processed/<name>/semantic_interaction.parquet",
            "layout": "sem_int_000..(d-1)=|u-v|, sem_int_d..(2d-1)=u*v",
            "config_gate": "features.semantic.save_interaction_vector",
        },
        "features": features,
    }
    out_dir = cfg.paths.reports / "phase2b"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_dictionary.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    log.info(f"wrote {out_dir / 'feature_dictionary.json'} ({len(features)} features)")


def run_all(cfg: DataConfig | None = None, only: list[str] | None = None) -> dict[str, dict]:
    cfg = cfg or load_data_config()
    ensure_dirs(cfg)
    set_global_seed(cfg.seed)
    if not cfg.features.enabled:
        log.warning("features.enabled is false — nothing to do")
        return {}

    branches = cfg.features.branches
    need_sbert = branches.semantic or branches.rubric
    need_nlp = branches.entities or branches.rubric

    nlp = load_feature_nlp(cfg.features.ner.spacy_model) if need_nlp else None
    encoder = None
    cache_path = cfg.paths.processed / ".sbert_cache.npz"
    if need_sbert:
        encoder = semantic.SbertEncoder(
            cfg.features.sbert_model,
            batch_size=cfg.features.semantic.batch_size,
            normalize=cfg.features.semantic.normalize,
            log=log,
        )
        if cfg.features.semantic.use_cache:
            encoder.load_cache(cache_path)

    names = only or [n for n in sorted(cfg.datasets) if (cfg.paths.processed / n / "encoder.parquet").exists()]
    results: dict[str, dict] = {}
    enabled_branches: list[str] = []
    for name in names:
        sidecar = build_dataset(name, cfg, encoder, nlp)
        if sidecar:
            results[name] = sidecar
            enabled_branches = sidecar["branches_enabled"]

    if encoder is not None and cfg.features.semantic.use_cache:
        encoder.save_cache(cache_path)
    if enabled_branches:
        _write_dictionary(cfg, enabled_branches)
    return results


if __name__ == "__main__":
    import sys

    run_all(only=sys.argv[1:] or None)
