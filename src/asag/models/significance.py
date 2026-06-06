"""Phase 2D — paired-bootstrap significance of the head vs the trivial baseline.

The report makes a significance test "near-mandatory at Q2/Q3". This computes a
**paired bootstrap** of the headline-metric gap Δ = head − baseline on the
headline test split (the same split the summary figure reports):

* SemEval → ``test_ud`` (cross-domain, the headline figure),
* ASAP-SAS → ``test_ua`` (per-prompt QWK, averaged across prompts),
* k-fold datasets → the pooled out-of-fold predictions.

We fit once with a single representative seed (``cfg.seed``) and the tuned
hyperparameters, get **per-item** predictions for both the head and the baseline
on the *same* items (that pairing is what makes the test valid), then resample
item indices ``n_boot`` times. For ASAP-SAS we resample *within each prompt* and
average the per-prompt metric, mirroring how the headline QWK is defined.

Reported: the observed gap, a percentile confidence interval, and a one-sided
p-value ``P(Δ ≤ 0)`` — the bootstrap probability the head does **not** beat the
baseline. Everything is deterministic via ``numpy.random.default_rng(seed)``.
"""

from __future__ import annotations

import numpy as np

from asag.config import DataConfig, LightGBMCfg
from asag.models.data import Bundle, load_bundle, make_y
from asag.models.evaluate import fit_predict_arrays
from asag.models.metrics import compute_metrics
from asag.models.tasks import get_spec
from asag.utils.logging import get_logger

log = get_logger()

_LOWER_IS_BETTER = {"rmse", "mae"}

# A "group" is one (y_true, head_pred, base_pred) block scored independently and
# resampled within itself. Non-per-prompt datasets have a single group; ASAP-SAS
# has one group per prompt (their per-group metrics are averaged).
Group = tuple[np.ndarray, np.ndarray, np.ndarray]


def _headline_over_groups(groups: list[Group], headline: str,
                          which: int, idx_per_group: list[np.ndarray]) -> float:
    """Mean headline metric over groups, each scored on its resampled indices.

    ``which`` selects head (1) or baseline (2) from each ``(yt, head, base)``.
    """
    vals = []
    for (yt, *preds), idx in zip(groups, idx_per_group):
        m = compute_metrics(yt[idx], preds[which - 1][idx], (headline,))[headline]
        if np.isfinite(m):
            vals.append(m)
    return float(np.mean(vals)) if vals else float("nan")


def _collect_groups(bundle: Bundle, cfg: DataConfig,
                    head_params: LightGBMCfg | None) -> tuple[list[Group], str]:
    """Per-item (y_true, head_pred, base_pred) blocks on the headline split."""
    spec, df = bundle.spec, bundle.df
    seed = cfg.seed

    if spec.protocol == "kfold":
        finite = np.isfinite(make_y(df, bundle))
        folds = sorted(int(f) for f in df["fold"].unique() if int(f) >= 0)
        yt_a, gp_a, bp_a = [], [], []
        for f in folds:
            te = df[(df["fold"] == f) & finite]
            tr = df[(df["fold"] != f) & (df["fold"] >= 0) & finite]
            if tr.empty or te.empty:
                continue
            yt, gp, bp = fit_predict_arrays(tr, te, bundle, cfg, seed, head_params)
            yt_a.append(yt); gp_a.append(gp); bp_a.append(bp)
        if not yt_a:
            return [], "cv"
        return [(np.concatenate(yt_a), np.concatenate(gp_a), np.concatenate(bp_a))], "cv"

    # official_split — headline is the last (hardest/cross-domain) test split.
    split = spec.test_splits[-1]
    train = df[(df["split"] == "train") & np.isfinite(make_y(df, bundle))]
    test = df[(df["split"] == split) & np.isfinite(make_y(df, bundle))]
    if train.empty or test.empty:
        return [], split

    if spec.per_prompt:
        groups: list[Group] = []
        for p in sorted(test["question_id"].astype(str).unique()):
            tr = train[train["question_id"].astype(str) == p]
            te = test[test["question_id"].astype(str) == p]
            if tr.empty or te.empty:
                continue
            groups.append(fit_predict_arrays(tr, te, bundle, cfg, seed, head_params))
        return groups, split

    return [fit_predict_arrays(train, test, bundle, cfg, seed, head_params)], split


def bootstrap_groups(groups: list[Group], headline: str, n_boot: int, ci: float,
                     seed: int, lower_is_better: bool = False) -> dict:
    """Pure paired-bootstrap over pre-computed prediction groups.

    Resamples item indices within each group ``n_boot`` times, averages the
    per-group headline metric, and reports the head−baseline gap with a CI and a
    one-sided p-value ``P(Δ ≤ 0)``. Kept free of I/O / LightGBM so it is unit
    testable on synthetic groups.

    **Degenerate-baseline fallback** — for an official-split *regression* dataset
    the baseline is a single constant (the training mean), and Pearson/Spearman of
    a constant is undefined (``nan``). The paired Δ would then be ``nan`` and
    uninformative. When the baseline metric is undefined we instead bootstrap the
    *head* metric against a null of 0 (``effect = "head_vs_zero"``) — i.e. "does
    the head reach a correlation reliably above zero" — which is the meaningful
    version of "does the head beat a zero-correlation constant predictor".
    """
    if not groups:
        return {"status": "empty"}
    sign = -1.0 if lower_is_better else 1.0
    full_idx = [np.arange(len(g[0])) for g in groups]
    obs_head = _headline_over_groups(groups, headline, 1, full_idx)
    obs_base = _headline_over_groups(groups, headline, 2, full_idx)

    rng = np.random.default_rng(seed)
    head_s = np.empty(n_boot, dtype=float)
    base_s = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = [rng.integers(0, len(g[0]), len(g[0])) for g in groups]
        head_s[b] = _headline_over_groups(groups, headline, 1, idx)
        base_s[b] = _headline_over_groups(groups, headline, 2, idx)

    lo_q = (1.0 - ci) / 2.0
    out = {
        "status": "ok",
        "metric": headline,
        "n_groups": len(groups),
        "n_items": int(sum(len(g[0]) for g in groups)),
        "n_boot": int(n_boot),
        "ci_level": ci,
        "head": round(obs_head, 4),
        "baseline": None if not np.isfinite(obs_base) else round(obs_base, 4),
    }

    if not np.isfinite(obs_base):
        # Constant baseline → metric undefined; test the head against a null of 0.
        eff = sign * head_s[np.isfinite(head_s)]
        ci_lo, ci_hi = np.quantile(eff, [lo_q, 1.0 - lo_q]) if eff.size else (np.nan, np.nan)
        out.update(
            baseline_degenerate=True,
            effect="head_vs_zero",
            delta_observed=round(float(sign * obs_head), 4),
            delta_boot_mean=round(float(np.mean(eff)), 4) if eff.size else float("nan"),
            ci_lo=round(float(ci_lo), 4),
            ci_hi=round(float(ci_hi), 4),
            p_value=float(np.mean(eff <= 0.0)) if eff.size else float("nan"),
            significant=bool(eff.size and ci_lo > 0.0),
        )
        return out

    deltas = sign * (head_s - base_s)
    finite = deltas[np.isfinite(deltas)]
    ci_lo, ci_hi = np.quantile(finite, [lo_q, 1.0 - lo_q]) if finite.size else (np.nan, np.nan)
    out.update(
        baseline_degenerate=False,
        effect="head_minus_baseline",
        delta_observed=round(float(sign * (obs_head - obs_base)), 4),
        delta_boot_mean=round(float(np.mean(finite)), 4) if finite.size else float("nan"),
        ci_lo=round(float(ci_lo), 4),
        ci_hi=round(float(ci_hi), 4),
        p_value=float(np.mean(finite <= 0.0)) if finite.size else float("nan"),
        significant=bool(finite.size and ci_lo > 0.0),
    )
    return out


def paired_bootstrap(bundle: Bundle, cfg: DataConfig,
                     head_params: LightGBMCfg | None = None) -> dict:
    """Paired-bootstrap the headline gap (head − baseline) on the headline split."""
    spec = bundle.spec
    sig = cfg.model.significance
    groups, split = _collect_groups(bundle, cfg, head_params)
    if not groups:
        return {"status": "empty", "split": split, "metric": spec.headline}

    result = bootstrap_groups(groups, spec.headline, sig.n_boot, sig.ci, sig.seed,
                              lower_is_better=spec.headline in _LOWER_IS_BETTER)
    result["split"] = split
    log.info(f"{bundle.name}: Δ{spec.headline}@{split}={result['delta_observed']:.4f} "
             f"CI[{result['ci_lo']:.4f},{result['ci_hi']:.4f}] p={result['p_value']:.4f}")
    return result


def significance_for(name: str, cfg: DataConfig,
                     head_params: LightGBMCfg | None = None) -> dict | None:
    """Convenience wrapper: load the bundle and run :func:`paired_bootstrap`."""
    bundle = load_bundle(name, cfg, get_spec(name))
    if bundle is None:
        return None
    return paired_bootstrap(bundle, cfg, head_params)
