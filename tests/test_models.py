"""Phase 2C — model-layer tests.

Pure/synthetic where possible; the LightGBM head tests ``importorskip`` the wheel
so the suite stays green without it (same pattern as test_features.py).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from asag.config import load_data_config
from asag.models import baselines, metrics
from asag.models.tasks import REGISTRY, get_spec


# ----------------------------- metrics -----------------------------------

def test_qwk_perfect_and_disagree():
    assert metrics.qwk([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    # constant prediction vs varied truth is defined (no agreement beyond chance)
    assert metrics.qwk([0, 1, 2], [1, 1, 1]) == pytest.approx(0.0)
    # both sides collapse to one label -> kappa undefined
    assert math.isnan(metrics.qwk([1, 1, 1], [1, 1, 1]))


def test_rmse_mae_known():
    assert metrics.rmse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(math.sqrt(4 / 3))
    assert metrics.mae([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(2 / 3)


def test_macro_f1_and_constant_corr():
    assert metrics.macro_f1([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)
    # correlation undefined when a side is constant
    assert math.isnan(metrics.pearson([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]))


def test_compute_metrics_empty_is_nan():
    out = metrics.compute_metrics([], [], ("qwk", "rmse"))
    assert all(math.isnan(v) for v in out.values())


# ------------------------------ tasks ------------------------------------

def test_registry_covers_six_datasets():
    assert set(REGISTRY) == {"semeval", "saf", "asap_sas", "mohler", "powergrading", "mindreading"}


def test_specs_are_internally_consistent():
    for name, spec in REGISTRY.items():
        assert spec.task_type in {"classification", "ordinal", "regression"}
        assert spec.target in {"label", "score"}
        assert (spec.target == "label") == (spec.task_type == "classification")
        assert spec.headline in spec.metrics
        if spec.protocol == "official_split":
            assert spec.test_splits
        assert get_spec(name) is spec


# ---------------------------- baselines ----------------------------------

def test_baseline_majority_and_mean():
    maj = baselines.fit_predict_baseline(np.array([0, 0, 1]), 4, "classification")
    assert maj.tolist() == [0, 0, 0, 0]
    mean = baselines.fit_predict_baseline(np.array([0.0, 4.0]), 2, "regression")
    assert mean.tolist() == [2.0, 2.0]
    med = baselines.fit_predict_baseline(np.array([0.0, 2.0, 2.0]), 1, "ordinal")
    assert med.tolist() == [2.0]


def test_constant_max():
    assert baselines.constant_max(np.array([1.0, 5.0, 3.0]), 2).tolist() == [5.0, 5.0]


# --------------------- LightGBM head (optional dep) ----------------------

@pytest.fixture(scope="module")
def cfg():
    return load_data_config()


def test_fusion_ordinal_with_nan_columns(cfg):
    pytest.importorskip("lightgbm")
    from asag.models.fusion import LgbmFusionHead

    rng = np.random.default_rng(0)
    n = 200
    signal = rng.integers(0, 3, size=n).astype(float)        # grades 0..2
    X = pd.DataFrame({
        "feat_signal": signal + rng.normal(0, 0.1, n),       # informative
        "feat_nan": np.full(n, np.nan),                       # all-NaN, NaN-native
        "feat_noise": rng.normal(0, 1, n),
    })
    head = LgbmFusionHead("ordinal", cfg.model.lightgbm, seed=42).fit(X, signal)
    pred = head.predict(X)
    assert pred.shape == (n,)
    assert set(np.unique(pred)).issubset({0.0, 1.0, 2.0})    # rounded + clipped to range
    assert (pred == np.rint(pred)).all()


def test_fusion_classification_returns_known_codes(cfg):
    pytest.importorskip("lightgbm")
    from asag.models.fusion import LgbmFusionHead

    rng = np.random.default_rng(1)
    n = 120
    y = rng.integers(0, 2, size=n).astype(float)
    X = pd.DataFrame({"a": y + rng.normal(0, 0.05, n), "b": rng.normal(0, 1, n)})
    head = LgbmFusionHead("classification", cfg.model.lightgbm, seed=7).fit(X, y)
    pred = head.predict(X)
    assert set(np.unique(pred)).issubset({0.0, 1.0})
