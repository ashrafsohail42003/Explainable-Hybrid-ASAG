"""Lexical-overlap and length features (``lex_*``, ``len_*``).

Operates on the **feature view** (``*_feat`` / ``*_feat_neg``) — lemmatized,
lowercased, stopword-stripped tokens — so overlap reflects content words, not
function words. Reference-dependent columns are ``NaN`` when the row has no
reference answer (empty ``reference_answer_feat``).
"""

from __future__ import annotations

import pandas as pd

from asag.features.text_utils import (
    NAN,
    dice,
    jaccard,
    log_ratio,
    ngram_set,
    overlap_precision,
    overlap_recall,
    safe_divide,
    tokens,
)

REF_DEPENDENT = [
    "lex_token_overlap",
    "lex_token_overlap_recall",
    "lex_token_overlap_prec",
    "lex_token_dice",
    "lex_bigram_overlap",
    "lex_trigram_overlap",
    "lex_content_word_overlap_neg",
    "len_ratio_sr",
    "len_logratio_sr",
]
ALWAYS = ["len_student_tokens", "len_student_chars", "len_student_uniq_ratio"]
COLUMNS = REF_DEPENDENT + ALWAYS


def compute_lexical(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    rows: list[dict] = []
    for s_feat, r_feat, s_neg, r_neg, s_enc in zip(
        df["student_answer_feat"], df["reference_answer_feat"],
        df["student_answer_feat_neg"], df["reference_answer_feat_neg"],
        df["student_answer_enc"],
    ):
        s = tokens(s_feat)
        r = tokens(r_feat)
        s_set, r_set = set(s), set(r)
        n_s = len(s)

        rec = {
            "len_student_tokens": float(n_s),
            "len_student_chars": float(len(s_enc) if isinstance(s_enc, str) else 0),
            "len_student_uniq_ratio": safe_divide(len(s_set), n_s),
        }
        if not r:  # no reference -> reference-dependent features undefined
            rec.update({c: NAN for c in REF_DEPENDENT})
        else:
            sn_set = set(tokens(s_neg))
            rn_set = set(tokens(r_neg))
            rec.update({
                "lex_token_overlap": jaccard(s_set, r_set),
                "lex_token_overlap_recall": overlap_recall(s_set, r_set),
                "lex_token_overlap_prec": overlap_precision(s_set, r_set),
                "lex_token_dice": dice(s_set, r_set),
                "lex_bigram_overlap": jaccard(ngram_set(s, 2), ngram_set(r, 2)),
                "lex_trigram_overlap": jaccard(ngram_set(s, 3), ngram_set(r, 3)),
                "lex_content_word_overlap_neg": jaccard(sn_set, rn_set),
                "len_ratio_sr": safe_divide(n_s, len(r)),
                "len_logratio_sr": log_ratio(n_s, len(r)),
            })
        rows.append(rec)

    return pd.DataFrame(rows, columns=COLUMNS, index=df.index)
