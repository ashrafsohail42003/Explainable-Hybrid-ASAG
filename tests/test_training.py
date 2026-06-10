"""Phase 2D — rigorous-training tests (HPO, paired bootstrap, IAA ceiling).

Pure where possible. The Optuna / LightGBM paths ``importorskip`` their wheels so
the suite stays green without them (same pattern as test_models.py).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from asag.config import load_data_config
from asag.models import ceiling, significance
from asag.models.data import Bundle
from asag.models.tasks import get_spec


@pytest.fixture(scope="module")
def cfg():
    return load_data_config()


def _synthetic_bundle(name: str, n: int = 160, seed: int = 0) -> Bundle:
    """A kfold regression bundle with a learnable signal (for HPO / array tests)."""
    rng = np.random.default_rng(seed)
    score = rng.integers(0, 6, size=n).astype(float)
    df = pd.DataFrame({
        # multiple question groups so grouped (leave-questions-out) inner CV is well-defined
        "question_id": [f"q{i % 16}" for i in range(n)],
        "score": score,
        "label": "",
        "dataset": name,
        "domain": "x",
        "split": "all",
        "fold": np.tile(np.arange(5), n // 5 + 1)[:n],
        "feat_signal": score + rng.normal(0, 0.3, n),
        "feat_noise": rng.normal(0, 1, n),
        "feat_nan": np.full(n, np.nan),
    })
    spec = get_spec("mohler")   # regression / kfold / pearson headline
    return Bundle(name=name, df=df, feature_cols=["feat_signal", "feat_noise", "feat_nan"],
                  spec=spec, label_vocab={})


# ------------------------------ HPO --------------------------------------

def test_hpo_returns_valid_config(cfg):
    pytest.importorskip("lightgbm")
    pytest.importorskip("optuna")
    from asag.config import LightGBMCfg
    from asag.models.hpo import SEARCH_SPACE, tune_dataset

    cfg2 = cfg.model_copy(deep=True)
    cfg2.model.hpo.n_trials = 4
    cfg2.model.hpo.inner_folds = 3
    bundle = _synthetic_bundle("synth")
    tuned, summary = tune_dataset(bundle, cfg2)

    assert isinstance(tuned, LightGBMCfg)
    assert summary["status"] == "ok"
    assert summary["validation"] == "inner_cv"      # no dev split → inner CV
    assert summary["n_trials"] == 4
    # tuned values stay inside the documented search space
    assert SEARCH_SPACE["num_leaves"]["low"] <= tuned.num_leaves <= SEARCH_SPACE["num_leaves"]["high"]
    assert 0.6 <= tuned.subsample <= 1.0


def test_hpo_disabled_falls_back_to_defaults(cfg):
    from asag.models.hpo import tune_dataset

    cfg2 = cfg.model_copy(deep=True)
    cfg2.model.hpo.enabled = False
    tuned, summary = tune_dataset(_synthetic_bundle("synth"), cfg2)
    assert summary["status"] == "skipped"
    assert tuned == cfg2.model.lightgbm


# ------------------------ paired bootstrap -------------------------------

def test_bootstrap_detects_clear_improvement():
    # head predicts perfectly; baseline is the constant median → QWK 1.0 vs 0.0
    yt = np.tile(np.arange(5), 30).astype(float)
    head = yt.copy()
    base = np.full_like(yt, 2.0)
    out = significance.bootstrap_groups([(yt, head, base)], "qwk",
                                        n_boot=400, ci=0.95, seed=42)
    assert out["delta_observed"] == pytest.approx(1.0, abs=1e-6)
    assert out["ci_lo"] > 0.0 and out["significant"]
    assert out["p_value"] < 0.01


def test_bootstrap_no_difference_is_not_significant():
    yt = np.tile(np.arange(5), 30).astype(float)
    pred = yt.copy()
    out = significance.bootstrap_groups([(yt, pred, pred)], "qwk",
                                        n_boot=400, ci=0.95, seed=42)
    assert out["delta_observed"] == pytest.approx(0.0, abs=1e-9)
    assert not out["significant"]
    assert out["p_value"] >= 0.5          # identical preds → no evidence of a gain


def test_bootstrap_is_deterministic():
    yt = np.tile(np.arange(4), 25).astype(float)
    head, base = yt.copy(), np.full_like(yt, 1.0)
    a = significance.bootstrap_groups([(yt, head, base)], "qwk", 300, 0.95, seed=7)
    b = significance.bootstrap_groups([(yt, head, base)], "qwk", 300, 0.95, seed=7)
    assert a == b


def test_cluster_bootstrap_widens_ci_vs_item_level():
    # Errors are correlated within a question (whole questions are right or wrong),
    # so i.i.d. item resampling fakes independence and yields a too-narrow CI; the
    # honest cluster (block) bootstrap over question_id must be wider on the SAME data.
    rng = np.random.default_rng(0)
    n_q, per_q = 12, 25
    qlabel = rng.integers(0, 5, n_q)
    yt = np.repeat(qlabel, per_q).astype(float)
    correct_q = rng.random(n_q) < 0.6                       # block-correlated errors
    head = np.where(np.repeat(correct_q, per_q), yt, (yt + 1) % 5).astype(float)
    base = np.full_like(yt, 2.0)
    qid = np.repeat(np.arange(n_q), per_q).astype(str)

    item = significance.bootstrap_groups([(yt, head, base)], "qwk",
                                         500, 0.95, seed=1)
    clus = significance.bootstrap_groups([(yt, head, base)], "qwk",
                                         500, 0.95, seed=1, clusters=[qid])
    assert item["resample_unit"] == "item" and item["n_clusters"] is None
    assert clus["resample_unit"] == "question" and clus["n_clusters"] == n_q
    assert (clus["ci_hi"] - clus["ci_lo"]) > (item["ci_hi"] - item["ci_lo"])


def test_holm_bonferroni_orders_and_corrects():
    # Holm adjusted p = monotone non-decreasing in rank; smallest gets (m)*p, and a
    # borderline raw-significant p can lose significance after family correction.
    p = {"a": 0.001, "b": 0.04, "c": 0.20, "d": 0.30}
    out = significance.holm_bonferroni(p, alpha=0.05)
    assert out["a"]["significant_holm"] and out["a"]["p_holm"] == pytest.approx(0.004)
    assert not out["b"]["significant_holm"]          # 0.04 raw-sig, killed by Holm
    adj = [out[k]["p_holm"] for k in ["a", "b", "c", "d"]]
    assert adj == sorted(adj)                         # monotone non-decreasing
    assert all(v <= 1.0 for v in adj)


def test_bootstrap_degenerate_baseline_falls_back_to_head_vs_zero():
    # constant baseline → Pearson(baseline) is nan; the Δ test is ill-posed, so we
    # fall back to a one-sample CI on the head correlation vs a null of 0.
    rng = np.random.default_rng(0)
    yt = rng.normal(0, 1, 200)
    head = yt + rng.normal(0, 0.3, 200)        # strongly correlated head
    base = np.full_like(yt, yt.mean())          # constant mean baseline
    out = significance.bootstrap_groups([(yt, head, base)], "pearson",
                                        n_boot=400, ci=0.95, seed=1)
    assert out["baseline_degenerate"] and out["effect"] == "head_vs_zero"
    assert out["baseline"] is None
    assert out["ci_lo"] > 0.0 and out["significant"]   # head correlation reliably > 0


# --------------------------- IAA ceiling ---------------------------------

def test_ceiling_unavailable_for_non_asap():
    for name in ("mohler", "saf", "mindreading", "semeval", "powergrading"):
        out = ceiling.ceiling_for(name)
        assert out["status"] == "unavailable"
        assert out["reason"]


def test_asap_ceiling_when_raw_present(cfg):
    ds = cfg.datasets["asap_sas"]
    if not (cfg.paths.raw / ds.raw_subdir / "train.tsv").exists():
        pytest.skip("ASAP-SAS raw TSVs not present")
    out = ceiling.asap_sas_ceiling(cfg)
    assert out["status"] == "ok"
    assert 0.0 < out["macro_qwk"] <= 1.0
    assert "test" not in out["splits_used"]      # Score2 withheld on test


# ------------------- regularization plumb-through ------------------------

def test_subsample_makes_head_seed_sensitive(cfg):
    pytest.importorskip("lightgbm")
    from asag.models.fusion import LgbmFusionHead

    rng = np.random.default_rng(0)
    n = 300
    y = rng.normal(0, 1, n)
    X = pd.DataFrame({"a": y + rng.normal(0, 0.5, n), "b": rng.normal(0, 1, n)})

    reg = cfg.model.lightgbm.model_copy(update={"subsample": 0.7, "colsample_bytree": 0.8})
    p1 = LgbmFusionHead("regression", reg, seed=1).fit(X, y).predict(X)
    p2 = LgbmFusionHead("regression", reg, seed=2).fit(X, y).predict(X)
    assert not np.allclose(p1, p2)               # bagging RNG now depends on the seed

    # default (subsample=1.0) stays deterministic — Phase 2C behavior is preserved
    d1 = LgbmFusionHead("regression", cfg.model.lightgbm, seed=1).fit(X, y).predict(X)
    d2 = LgbmFusionHead("regression", cfg.model.lightgbm, seed=2).fit(X, y).predict(X)
    assert np.allclose(d1, d2)
