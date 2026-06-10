"""Phase B — three-way comparison tests (pure core; no torch / lightgbm)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asag.models.data import Bundle
from asag.models.neural_compare import neural_only_headline
from asag.models.tasks import get_spec


def _bundle(name: str, df: pd.DataFrame, vocab: dict | None = None) -> Bundle:
    return Bundle(name=name, df=df.reset_index(drop=True), feature_cols=[],
                  spec=get_spec(name), label_vocab=vocab or {})


def test_neural_only_absent_when_no_columns():
    df = pd.DataFrame({"question_id": ["q"], "score": [1.0], "label": [""],
                       "dataset": ["mohler"], "domain": ["cs"], "split": ["all"], "fold": [0]})
    assert neural_only_headline(_bundle("mohler", df))["status"] == "absent"


def test_neural_only_kfold_regression():
    # neural_score perfectly tracks the gold score on OOF rows → Pearson ≈ 1.
    rng = np.random.default_rng(0)
    n = 60
    score = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "question_id": [f"q{i%6}" for i in range(n)],
        "score": score, "label": "", "dataset": "mohler", "domain": "cs",
        "split": "all", "fold": [i % 5 for i in range(n)],
        "neural_score": score + rng.normal(0, 0.01, n),   # near-perfect
        "neural_pred": np.round(score),
    })
    out = neural_only_headline(_bundle("mohler", df))
    assert out["status"] == "ok" and out["split"] == "cv"
    assert out["headline"]["metric"] == "pearson"
    assert out["headline"]["mean"] > 0.95


def test_neural_only_official_classification_uses_headline_split():
    # macro-F1 read from neural_pred on the headline (last) test split only.
    spec = get_spec("semeval")
    split = spec.test_splits[-1]
    vocab = {"correct": 0, "incorrect": 1}
    labels = (["correct", "incorrect"] * 20)
    n = len(labels)
    codes = np.array([vocab[l] for l in labels], dtype=float)
    df = pd.DataFrame({
        "question_id": [f"q{i%4}" for i in range(n)],
        "score": np.nan, "label": labels, "dataset": "semeval", "domain": "sci",
        "split": ([ "train"] * (n // 2)) + ([split] * (n - n // 2)),
        "fold": -1,
        "neural_score": codes,
        "neural_pred": codes,                              # perfect on the test split
    })
    out = neural_only_headline(_bundle("semeval", df, vocab))
    assert out["status"] == "ok" and out["split"] == split
    assert out["headline"]["metric"] == "macro_f1"
    assert out["headline"]["mean"] == 1.0
