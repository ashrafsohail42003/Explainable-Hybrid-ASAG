"""Named-entity overlap features (``ner_*``).

Runs spaCy NER (re-enabled; parser stays off) on the **encoder view** (raw
casing/punctuation, which NER needs). Entity *mentions* are compared by
lowercased surface text. Reference-dependent columns are ``NaN`` without a
reference.
"""

from __future__ import annotations

import pandas as pd

from asag.features.text_utils import NAN, jaccard, safe_divide

ALWAYS = ["ner_student_count", "ner_student_uniq_types"]
REF_DEPENDENT = ["ner_overlap_count", "ner_overlap_jaccard", "ner_ref_recall"]
COLUMNS = ALWAYS + REF_DEPENDENT


def compute_entities(df: pd.DataFrame, cfg, nlp) -> pd.DataFrame:
    students = df["student_answer_enc"].fillna("").astype(str).tolist()
    refs = df["reference_answer_enc"].fillna("").astype(str).tolist()
    batch = cfg.features.semantic.batch_size

    rows: list[dict] = []
    for s_doc, r_doc, r_text in zip(
        nlp.pipe(students, batch_size=batch),
        nlp.pipe(refs, batch_size=batch),
        refs,
    ):
        s_ents = {e.text.lower() for e in s_doc.ents}
        rec = {
            "ner_student_count": float(len(list(s_doc.ents))),
            "ner_student_uniq_types": float(len({e.label_ for e in s_doc.ents})),
        }
        if not r_text.strip():  # no reference
            rec.update({c: NAN for c in REF_DEPENDENT})
        else:
            r_ents = {e.text.lower() for e in r_doc.ents}
            rec.update({
                "ner_overlap_count": float(len(s_ents & r_ents)),
                "ner_overlap_jaccard": jaccard(s_ents, r_ents),
                "ner_ref_recall": safe_divide(len(s_ents & r_ents), len(r_ents)),
            })
        rows.append(rec)

    return pd.DataFrame(rows, columns=COLUMNS, index=df.index)
