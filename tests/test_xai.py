"""Phase 2F — XAI tests.

Pure logic is tested with light fakes (no SBERT/spaCy/LightGBM needed); the
native-TreeSHAP additivity test ``importorskip``s the LightGBM wheel.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from asag.config import load_data_config
from asag.xai import concept_attribution, saf_validation
from asag.xai.common import global_importance


@pytest.fixture(scope="module")
def cfg():
    return load_data_config()


# ---- fakes so the pure logic needs no SBERT / spaCy ----------------------

class FakeEncoder:
    """One-hot encoder: cosine == 1 for identical strings, else 0."""
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.dim = 256

    def embed(self, texts):
        M = np.zeros((len(texts), self.dim), dtype=float)
        for i, t in enumerate(texts):
            self.vocab.setdefault(t, len(self.vocab))
            M[i, self.vocab[t] % self.dim] = 1.0
        return M


class _Doc:
    def __init__(self, text):
        self._s = [s for s in text.split(". ") if s.strip()]

    @property
    def sents(self):
        return [type("S", (), {"text": s})() for s in self._s]


class FakeNLP:
    def pipe(self, texts, **kw):
        for t in texts:
            yield _Doc(t)


# ------------------------- concept attribution ----------------------------

def test_concept_coverage_marks_covered_and_missed():
    refs = ["alpha. beta. gamma"]
    students = ["beta"]                          # matches exactly one concept
    cov = concept_attribution.concept_coverage(refs, students, FakeEncoder(), FakeNLP(), tau=0.5)
    assert len(cov) == 1 and len(cov[0]) == 3
    covered = [c["concept"] for c in cov[0] if c["covered"]]
    assert covered == ["beta"]
    assert all(c["similarity"] in (0.0, 1.0) for c in cov[0])


def test_concept_coverage_empty_reference_yields_no_concepts():
    cov = concept_attribution.concept_coverage([""], ["anything"], FakeEncoder(), FakeNLP(), tau=0.5)
    assert cov == [[]]


# --------------------------- SAF validation -------------------------------

def test_signal_stats_detects_monotone_alignment():
    rng = np.random.default_rng(0)
    verdict = np.array((["Incorrect"] * 20) + (["Partially correct"] * 20) + (["Correct"] * 20))
    ordinal = np.array([{"Incorrect": 0, "Partially correct": 1, "Correct": 2}[v] for v in verdict], float)
    sig = ordinal + rng.normal(0, 0.1, ordinal.size)     # signal rises with the verdict
    out = saf_validation._signal_stats(sig, ordinal, verdict)
    assert out["monotonic"] is True
    assert out["spearman_vs_verdict"] > 0.8
    assert out["auc_correct_vs_rest"] > 0.9


def test_signal_stats_no_alignment_is_flat():
    rng = np.random.default_rng(1)
    verdict = np.array((["Incorrect"] * 20) + (["Correct"] * 20))
    ordinal = np.array([0] * 20 + [2] * 20, float)
    sig = rng.normal(0, 1, 40)                            # pure noise, no alignment
    out = saf_validation._signal_stats(sig, ordinal, verdict)
    assert abs(out["spearman_vs_verdict"]) < 0.5


# ------------------------- native TreeSHAP --------------------------------

def test_pred_contrib_additivity_and_global(cfg):
    pytest.importorskip("lightgbm")
    from asag.xai.common import shaped_contribs
    from asag.models.fusion import LgbmFusionHead

    rng = np.random.default_rng(0)
    n = 300
    y = rng.normal(0, 1, n)
    X = pd.DataFrame({"a": y + rng.normal(0, 0.3, n), "b": rng.normal(0, 1, n),
                      "c": np.full(n, np.nan)})
    head = LgbmFusionHead("regression", cfg.model.lightgbm, seed=0).fit(X, y)

    arr, k = head.pred_contrib(X)
    assert k == 1 and arr.shape == (n, X.shape[1] + 1)
    # TreeSHAP additivity: contributions + base == raw model output
    assert np.allclose(arr.sum(axis=1), head.model.predict(X), atol=1e-6)

    contribs = shaped_contribs(head, X, X.shape[1])
    assert contribs.shape == (n, X.shape[1])
    imp = global_importance(contribs)
    assert imp.shape == (X.shape[1],)
    assert imp[0] > imp[2]                                # 'a' (signal) beats all-NaN 'c'
