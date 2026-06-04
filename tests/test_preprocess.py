"""Phase 2A — preprocessing transform tests.

Covers the encoder-view lossless invariant and the new negation-scope cues on
the feature view. The spaCy-dependent tests skip cleanly when the model is not
installed so a bare clone still runs the suite.
"""

from __future__ import annotations

import pytest

from asag.data.preprocess import FeatureViewOptions, encoder_view, feature_view_batch

NEGATORS = ("not", "no", "never", "n't", "without", "cannot")


def test_encoder_view_preserves_case_and_punct():
    """Encoder view = NFKC + whitespace only; casing/punctuation untouched."""
    src = "The   TCP/IP Router does NOT forward packets!"
    out = encoder_view(src)
    assert out == "The TCP/IP Router does NOT forward packets!"
    # NFKC folds compatibility forms (e.g. fullwidth) but keeps casing.
    assert encoder_view("ＡＢＣ") == "ABC"
    assert encoder_view(None) == ""


@pytest.fixture(scope="module")
def nlp():
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        pytest.skip("en_core_web_sm not installed")


def _opts(negation_scope: bool, window: int = 4) -> FeatureViewOptions:
    return FeatureViewOptions(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        preserve_negators=NEGATORS,
        negation_scope=negation_scope,
        negation_window=window,
    )


def test_negation_scope_marks_window(nlp):
    plain, marked = feature_view_batch(
        ["The router does not forward packets"], nlp, _opts(True)
    )
    # negators are preserved as tokens in both views
    assert "not" in plain[0]
    # the negator and the tokens in its scope are prefixed in the marked view
    assert "neg_not" in marked[0]
    assert "neg_forward" in marked[0] or "neg_packet" in marked[0]
    # plain view carries no marking
    assert "neg_" not in plain[0]


def test_negation_scope_stops_at_clause_boundary(nlp):
    plain, marked = feature_view_batch(
        ["It does not work, the cable is fine"], nlp, _opts(True)
    )
    # tokens after the comma (clause boundary) must NOT be marked
    assert "neg_cable" not in marked
    assert "neg_fine" not in marked
    # something inside the first clause IS marked
    assert "neg_" in marked[0]


def test_negation_scope_disabled_is_plain(nlp):
    plain, marked = feature_view_batch(
        ["The router does not forward packets"], nlp, _opts(False)
    )
    assert marked == plain
    assert "neg_" not in marked[0]
