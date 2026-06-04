"""Shared helpers for the Phase 2B feature branches.

Feature-view text (``*_feat`` / ``*_feat_neg``) is already lemmatized, lowercased
and whitespace-joined by Phase 2A, so tokenization here is a plain ``split()``.
Set-overlap helpers return ``NaN`` (not 0) when a denominator is undefined — the
caller relies on that to surface "no reference" rows for the GBM head.
"""

from __future__ import annotations

import math

import numpy as np

NAN = float("nan")


def tokens(text: str) -> list[str]:
    """Whitespace tokens of an already-normalized feature-view string."""
    if not isinstance(text, str):
        return []
    return text.split()


def ngram_set(toks: list[str], n: int) -> set[tuple[str, ...]]:
    if n <= 1:
        return set(toks)
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def safe_divide(num: float, den: float) -> float:
    """``num/den`` but ``NaN`` when ``den == 0`` (undefined, not zero)."""
    return num / den if den else NAN


def jaccard(a: set, b: set) -> float:
    """|a∩b| / |a∪b|; NaN when both sides empty (undefined)."""
    if not a and not b:
        return NAN
    return safe_divide(len(a & b), len(a | b))


def overlap_recall(student: set, reference: set) -> float:
    """|S∩R| / |R| — how much of the reference the student covered."""
    return safe_divide(len(student & reference), len(reference))


def overlap_precision(student: set, reference: set) -> float:
    return safe_divide(len(student & reference), len(student))


def dice(a: set, b: set) -> float:
    if not a and not b:
        return NAN
    return safe_divide(2 * len(a & b), len(a) + len(b))


def log_ratio(num_len: int, den_len: int) -> float:
    """log((num+1)/(den+1)) — symmetric, defined even when a side is empty."""
    return math.log((num_len + 1) / (den_len + 1))


def cosine_rows(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Row-wise cosine for two (n, d) matrices; 0 when a row is all-zero."""
    un = np.linalg.norm(u, axis=1)
    vn = np.linalg.norm(v, axis=1)
    denom = un * vn
    dot = np.einsum("ij,ij->i", u, v)
    out = np.zeros_like(dot, dtype=np.float64)
    nz = denom > 0
    out[nz] = dot[nz] / denom[nz]
    return out


def load_feature_nlp(spacy_model: str):
    """spaCy pipe for the feature branches: NER on, parser off, sentencizer added.

    Distinct from ``preprocess._load_spacy`` (which disables NER). We keep the
    parser off (slow, low-value per the report) but add the rule-based
    ``sentencizer`` so ``doc.sents`` works for rubric concept splitting.
    """
    import spacy

    try:
        nlp = spacy.load(spacy_model, disable=["parser"])
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{spacy_model}' not installed. Run: python -m spacy download {spacy_model}"
        ) from e
    if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)
    return nlp
