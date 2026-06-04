"""Negation-cue and contradiction features (``neg_*``).

Derived from the Phase 2A negation-scope view: tokens inside a negator's scope
carry a ``neg_`` prefix in ``*_feat_neg``. We count them and compare student vs
reference polarity. ``neg_overlap_delta`` measures how much the scope marking
shifts lexical overlap (a proxy for "same words, opposite polarity").
"""

from __future__ import annotations

import pandas as pd

from asag.features.text_utils import NAN, jaccard, tokens

ALWAYS = ["neg_student_count", "neg_student_present"]
REF_DEPENDENT = ["neg_reference_count", "neg_polarity_mismatch", "neg_overlap_delta"]
COLUMNS = ALWAYS + REF_DEPENDENT


def _neg_count(neg_text: str) -> int:
    return sum(1 for t in tokens(neg_text) if t.startswith("neg_"))


def compute_negation(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    rows: list[dict] = []
    for s_feat, r_feat, s_neg, r_neg in zip(
        df["student_answer_feat"], df["reference_answer_feat"],
        df["student_answer_feat_neg"], df["reference_answer_feat_neg"],
    ):
        s_neg_count = _neg_count(s_neg)
        s_present = s_neg_count > 0
        rec = {
            "neg_student_count": float(s_neg_count),
            "neg_student_present": float(s_present),
        }
        if not tokens(r_feat):  # no reference
            rec.update({c: NAN for c in REF_DEPENDENT})
        else:
            r_neg_count = _neg_count(r_neg)
            r_present = r_neg_count > 0
            plain = jaccard(set(tokens(s_feat)), set(tokens(r_feat)))
            marked = jaccard(set(tokens(s_neg)), set(tokens(r_neg)))
            delta = (plain - marked) if (plain == plain and marked == marked) else NAN
            rec.update({
                "neg_reference_count": float(r_neg_count),
                "neg_polarity_mismatch": float(s_present ^ r_present),
                "neg_overlap_delta": delta,
            })
        rows.append(rec)

    return pd.DataFrame(rows, columns=COLUMNS, index=df.index)
