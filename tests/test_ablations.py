"""Phase 3 — ablation tests (pure feature-partitioning logic, no deps)."""

from __future__ import annotations

from asag.models import ablations

# a representative feature_cols list covering every branch prefix (incl. neural D)
FEATS = [
    "lex_token_overlap", "lex_content_word_overlap_neg", "len_student_chars",
    "tfidf_cosine", "neg_student_count", "neg_polarity_mismatch",
    "ner_overlap_jaccard", "sem_cosine", "sem_abs_diff_mean",
    "rub_mean_maxsim", "rub_coverage_at_tau",
    "neural_score", "neural_pred",
]


def test_branch_partition_is_complete_and_disjoint():
    g = ablations._groups(FEATS)
    # every feature lands in exactly one branch (A/B/C/D)
    union = g["A"] + g["B"] + g["C"] + g["D"]
    assert sorted(union) == sorted(FEATS)
    assert g["A"] == ["sem_cosine", "sem_abs_diff_mean"]
    assert g["C"] == ["rub_mean_maxsim", "rub_coverage_at_tau"]
    assert g["D"] == ["neural_score", "neural_pred"]
    assert set(g["B"]).isdisjoint(g["A"] + g["C"] + g["D"])


def test_neural_branch_d_variants():
    # -D drops only the neural features; only-D keeps only them; branch_of tags them D.
    assert ablations.branch_of("neural_score") == "D"
    minus_d = ablations.variant_cols(FEATS, "-D")
    assert "neural_score" not in minus_d and "neural_pred" not in minus_d
    assert "sem_cosine" in minus_d and "lex_token_overlap" in minus_d
    assert ablations.variant_cols(FEATS, "only-D") == ["neural_score", "neural_pred"]


def test_drop_variants_remove_the_right_branch():
    assert all(not c.startswith("sem_") for c in ablations.variant_cols(FEATS, "-A"))
    assert all(not c.startswith(("lex_", "len_", "tfidf_", "neg_", "ner_"))
               for c in ablations.variant_cols(FEATS, "-B"))
    assert all(not c.startswith("rub_") for c in ablations.variant_cols(FEATS, "-C"))
    # full keeps everything
    assert ablations.variant_cols(FEATS, "full") == FEATS


def test_only_variants_keep_one_branch():
    assert ablations.variant_cols(FEATS, "only-A") == ["sem_cosine", "sem_abs_diff_mean"]
    assert ablations.variant_cols(FEATS, "only-C") == ["rub_mean_maxsim", "rub_coverage_at_tau"]
    assert all(c.startswith(("lex_", "len_", "tfidf_", "neg_", "ner_"))
               for c in ablations.variant_cols(FEATS, "only-B"))


def test_negation_ablation_drops_neg_cues_only():
    kept = ablations.variant_cols(FEATS, "-neg")
    assert "neg_student_count" not in kept and "neg_polarity_mismatch" not in kept
    assert "lex_content_word_overlap_neg" not in kept     # negation-scope lexical cue
    assert "lex_token_overlap" in kept and "sem_cosine" in kept   # untouched
    assert "rub_mean_maxsim" in kept


def test_branch_of():
    assert ablations.branch_of("sem_cosine") == "A"
    assert ablations.branch_of("rub_mean_maxsim") == "C"
    assert ablations.branch_of("lex_token_overlap") == "B"
    assert ablations.branch_of("neg_student_count") == "B"
