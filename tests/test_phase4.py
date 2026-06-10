"""Phase 4 tests — grouped CV, leakage controls, LODO binarization, calibration.

Pure where possible; the lightgbm-backed checks ``importorskip`` the wheel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asag.data.splits import make_grouped_kfold, make_stratified_kfold


def _toy(n_q=10, per_q=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_q):
        base = rng.integers(0, 5)
        for _ in range(per_q):
            rows.append({"question_id": f"q{q}", "score": float((base + rng.integers(0, 2)) % 5),
                         "label": "correct" if rng.random() > 0.5 else "incorrect"})
    return pd.DataFrame(rows)


# --------------------------- grouped CV ----------------------------------

def test_grouped_kfold_no_question_leak():
    df = _toy()
    fold = make_grouped_kfold(df, k=5, seed=42, group_col="question_id")
    for f in sorted(fold.unique()):
        tr_q = set(df.loc[fold != f, "question_id"])
        te_q = set(df.loc[fold == f, "question_id"])
        assert tr_q.isdisjoint(te_q), f"fold {f}: question leaked across train/test"


def test_grouped_kfold_covers_all_rows():
    df = _toy()
    fold = make_grouped_kfold(df, k=5, seed=42)
    assert (fold >= 0).all() and set(fold.unique()) == {0, 1, 2, 3, 4}


def test_grouped_kfold_fewer_groups_than_k_falls_back():
    df = _toy(n_q=3, per_q=30)            # 3 groups, ask for k=5
    fold = make_grouped_kfold(df, k=5, seed=42)
    # still disjoint by question, just fewer folds
    for f in sorted(fold.unique()):
        tr_q = set(df.loc[fold != f, "question_id"])
        te_q = set(df.loc[fold == f, "question_id"])
        assert tr_q.isdisjoint(te_q)
    assert fold.nunique() <= 3


def test_stratified_kfold_does_leak_questions():
    """Documents the bug the fix removes: the legacy split shares questions."""
    df = _toy()
    fold = make_stratified_kfold(df, k=5, seed=42)
    leaked = any(
        not set(df.loc[fold != f, "question_id"]).isdisjoint(set(df.loc[fold == f, "question_id"]))
        for f in sorted(fold.unique()))
    assert leaked


# --------------------------- shortcut / prior ----------------------------

def test_question_shortcut_collapses_on_unseen_questions():
    from asag.models.baselines import question_shortcut_predict
    y = np.array([0.0, 0.0, 4.0, 4.0])
    qtr = ["a", "a", "b", "b"]
    # seen question -> recovers the per-question mean
    seen = question_shortcut_predict(y, qtr, ["a", "b"], "regression")
    assert seen[0] == pytest.approx(0.0) and seen[1] == pytest.approx(4.0)
    # unseen question -> global fallback (memorization shortcut dies)
    unseen = question_shortcut_predict(y, qtr, ["zzz"], "regression")
    assert unseen[0] == pytest.approx(np.mean(y))


def test_question_prior_loo_and_nan_for_unseen():
    from asag.models.data import question_prior
    from asag.models.tasks import get_spec
    tr = pd.DataFrame({"question_id": ["a", "a", "a"], "score": [1.0, 2.0, 3.0]})
    te = pd.DataFrame({"question_id": ["a", "zzz"], "score": [0.0, 0.0]})
    tr_p, te_p = question_prior(tr, te, get_spec("mohler"))
    # LOO: first 'a' row prior excludes itself -> mean(2,3)=2.5
    assert tr_p[0] == pytest.approx(2.5)
    # test: seen 'a' -> 2.0 ; unseen -> NaN
    assert te_p[0] == pytest.approx(2.0) and np.isnan(te_p[1])


def test_question_prior_nan_for_classification():
    from asag.models.data import question_prior
    from asag.models.tasks import get_spec
    tr = pd.DataFrame({"question_id": ["a"], "score": [np.nan], "label": ["correct"]})
    te = pd.DataFrame({"question_id": ["a"], "score": [np.nan], "label": ["correct"]})
    tr_p, te_p = question_prior(tr, te, get_spec("semeval"))
    assert np.isnan(tr_p).all() and np.isnan(te_p).all()


# --------------------------- LODO binarization ---------------------------

def test_lodo_binarize_label_and_score():
    from asag.models.lodo import binarize_target
    lab = pd.DataFrame({"label": ["correct", "incorrect", "partially_correct_incomplete"],
                        "score": [np.nan, np.nan, np.nan]})
    out = binarize_target(lab, "semeval")
    assert list(out) == [1.0, 0.0, 0.0]
    sc = pd.DataFrame({"label": ["", "", ""], "score": [0.0, 5.0, np.nan]})
    out2 = binarize_target(sc, "mohler")          # midpoint 2.5
    assert out2[0] == 0.0 and out2[1] == 1.0 and np.isnan(out2[2])


# --------------------------- calibration math ----------------------------

def test_ece_perfectly_calibrated_is_zero():
    from asag.models.robustness import expected_calibration_error
    # full-confidence predictions that are all correct -> ECE 0
    y = np.array([1, 1, 0, 0])
    proba = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    ece, _ = expected_calibration_error(y, proba, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_ece_overconfident_is_positive():
    from asag.models.robustness import expected_calibration_error
    # full confidence but half wrong -> ECE ~ 0.5
    y = np.array([1, 1, 0, 0])
    proba = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    ece, _ = expected_calibration_error(y, proba, n_bins=5)
    assert ece == pytest.approx(0.5, abs=1e-9)


def test_temperature_scaling_reduces_overconfident_ece():
    from asag.models.robustness import (expected_calibration_error,
                                        fit_temperature, temperature_scale)
    # Overconfident probabilities: the head always says class 1 at 0.8 confidence
    # but is only right 60% of the time → an 0.2 confidence gap to correct.
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.6).astype(int)                # correct (==1) 60% of rows
    proba = np.tile([0.2, 0.8], (n, 1))                  # constant 0.8 confidence in class 1
    t = fit_temperature(proba, y)
    ece_pre, _ = expected_calibration_error(y, proba)
    ece_post, _ = expected_calibration_error(y, temperature_scale(proba, t))
    assert t > 1.0                       # overconfidence → softening temperature
    assert ece_post < ece_pre            # calibration improves out-of-sample-style


def test_temperature_scale_identity_at_one():
    from asag.models.robustness import temperature_scale
    proba = np.array([[0.2, 0.8], [0.7, 0.3], [0.5, 0.5]])
    np.testing.assert_allclose(temperature_scale(proba, 1.0), proba, atol=1e-9)


def test_risk_coverage_monotone_in_confidence():
    from asag.models.robustness import risk_coverage
    y = np.array([1, 1, 0, 0])
    # most-confident two are correct, least-confident two are wrong
    proba = np.array([[0.1, 0.9], [0.2, 0.8], [0.45, 0.55], [0.55, 0.45]])
    rc = {d["coverage"]: d["accuracy"] for d in risk_coverage(y, proba, points=(1.0, 0.5))}
    assert rc[0.5] >= rc[1.0]   # keeping only confident predictions is at least as accurate
