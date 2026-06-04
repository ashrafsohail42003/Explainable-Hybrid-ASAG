"""Phase 2B — feature-branch tests.

Mostly synthetic frames so the suite runs without the real datasets. SBERT/spaCy
tests skip cleanly when the optional dependency or model is absent.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from asag.config import load_data_config
from asag.features import lexical, negation, tfidf
from asag.features.build import _merge_views


@pytest.fixture(scope="module")
def cfg():
    return load_data_config()


def _frame(students, references):
    """Build a minimal merged frame; feature view == lowercased token join."""
    def feat(x):
        return x.lower()
    return pd.DataFrame({
        "student_answer_feat": [feat(s) for s in students],
        "reference_answer_feat": [feat(r) for r in references],
        "student_answer_feat_neg": [feat(s) for s in students],
        "reference_answer_feat_neg": [feat(r) for r in references],
        "student_answer_enc": students,
        "reference_answer_enc": references,
    })


def test_lexical_overlap_toy():
    df = _frame(["router forward packet"], ["router forward packet data"])
    out = lexical.compute_lexical(df)
    assert out["lex_token_overlap"].iloc[0] == pytest.approx(0.75)
    assert out["lex_token_overlap_recall"].iloc[0] == pytest.approx(0.75)
    assert out["lex_token_overlap_prec"].iloc[0] == pytest.approx(1.0)
    assert out["len_student_tokens"].iloc[0] == 3.0


def test_lexical_empty_reference_is_nan():
    df = _frame(["router forward packet"], [""])
    out = lexical.compute_lexical(df)
    for c in lexical.REF_DEPENDENT:
        assert math.isnan(out[c].iloc[0]), c
    # student-only features remain finite
    assert out["len_student_tokens"].iloc[0] == 3.0
    assert out["len_student_uniq_ratio"].iloc[0] == pytest.approx(1.0)


def test_negation_counts_and_nan():
    df = _frame(["router neg_not neg_forward packet"], [""])
    out = negation.compute_negation(df)
    assert out["neg_student_count"].iloc[0] == 2.0
    assert out["neg_student_present"].iloc[0] == 1.0
    for c in negation.REF_DEPENDENT:
        assert math.isnan(out[c].iloc[0]), c


def test_negation_polarity_mismatch():
    df = _frame(["router neg_not forward"], ["router forward"])
    out = negation.compute_negation(df)
    # student has negation, reference does not -> XOR == 1
    assert out["neg_polarity_mismatch"].iloc[0] == 1.0
    assert out["neg_reference_count"].iloc[0] == 0.0


def test_tfidf_cosine_bounds(cfg):
    df = _frame(
        ["alpha beta gamma", "alpha beta gamma", "alpha beta gamma"],
        ["alpha beta gamma", "delta epsilon zeta", ""],
    )
    out = tfidf.compute_tfidf(df, cfg)
    assert out["tfidf_cosine"].iloc[0] == pytest.approx(1.0, abs=1e-6)   # identical
    assert out["tfidf_cosine"].iloc[1] == pytest.approx(0.0, abs=1e-6)   # disjoint
    assert math.isnan(out["tfidf_cosine"].iloc[2])                       # empty ref


def test_build_alignment_assert():
    enc = pd.DataFrame({"question_id": ["a", "b"]})
    feat = pd.DataFrame({"question_id": ["b", "a"]})  # different order
    with pytest.raises(ValueError, match="not aligned"):
        _merge_views("toy", enc, feat)


# --- optional-dependency integration tests --------------------------------

@pytest.fixture(scope="module")
def nlp():
    pytest.importorskip("spacy")
    from asag.features.text_utils import load_feature_nlp
    try:
        return load_feature_nlp("en_core_web_sm")
    except RuntimeError:
        pytest.skip("en_core_web_sm not installed")


@pytest.fixture(scope="module")
def encoder(cfg):
    pytest.importorskip("sentence_transformers")
    from asag.features.semantic import SbertEncoder
    return SbertEncoder(cfg.features.sbert_model, batch_size=8, normalize=True)


def test_semantic_shapes_and_range(cfg, encoder):
    from asag.features import semantic
    df = pd.DataFrame({
        "student_answer_enc": ["the router forwards packets", "no reference here"],
        "reference_answer_enc": ["a router forwards packets", ""],
    })
    cfg.features.semantic.save_interaction_vector = True
    scalars, dense = semantic.compute_semantic(df, cfg, encoder)
    cfg.features.semantic.save_interaction_vector = False
    assert -1.01 <= scalars["sem_cosine"].iloc[0] <= 1.01
    assert math.isnan(scalars["sem_cosine"].iloc[1])      # no reference
    assert dense.shape == (2, 2 * encoder.dim)
    assert np.isnan(dense[1]).all()                       # no-ref row is NaN


def test_rubric_coverage_toy(cfg, encoder, nlp):
    from asag.features import rubric
    df = pd.DataFrame({
        "student_answer_enc": ["TCP is reliable. It uses sequence numbers."],
        "reference_answer_enc": ["TCP is reliable. TCP uses sequence numbers."],
    })
    out = rubric.compute_rubric(df, cfg, encoder, nlp)
    assert out["rub_n_concepts"].iloc[0] == 2.0
    assert 0.0 <= out["rub_coverage_at_tau"].iloc[0] <= 1.0
    assert out["rub_max_maxsim"].iloc[0] >= out["rub_min_maxsim"].iloc[0]

    empty = pd.DataFrame({"student_answer_enc": ["anything"], "reference_answer_enc": [""]})
    out2 = rubric.compute_rubric(empty, cfg, encoder, nlp)
    assert math.isnan(out2["rub_n_concepts"].iloc[0])


def test_entities_smoke(cfg, nlp):
    from asag.features import entities
    df = pd.DataFrame({
        "student_answer_enc": ["Albert Einstein developed relativity"],
        "reference_answer_enc": [""],
    })
    out = entities.compute_entities(df, cfg, nlp)
    assert out["ner_student_count"].iloc[0] >= 0.0
    for c in entities.REF_DEPENDENT:
        assert math.isnan(out[c].iloc[0]), c
