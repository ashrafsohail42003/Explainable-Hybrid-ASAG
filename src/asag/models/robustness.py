"""Phase 4 — robustness, calibration, and selective prediction.

Three reviewer-facing stress tests the accuracy table cannot show:

* **Perturbations** — edit the student answer and measure how much the head's score
  moves. We recompute only the lexical + negation branches on the perturbed text
  (pure, per-row; semantic/rubric/ner/tfidf held at their original cached values —
  a *conservative* setting: the semantic branch could only add robustness). Tests:
    - ``length_pad``      — append off-topic filler. A length-robust grader barely moves.
    - ``paraphrase_drop`` — drop a fraction of content tokens (meaning ~kept, surface
      changed). Exposes over-reliance on exact lexical overlap.
    - ``negation_flip``   — negate the whole answer. The score *should* drop.
* **Calibration (ECE) + temperature scaling** — classification heads via
  ``predict_proba``; reliability diagram + expected calibration error, reported
  **before and after** post-hoc temperature scaling. The temperature ``T`` is fit
  by NLL grid-search on a held-out *validation* slice (never the test split), so
  the post-calibration ECE is an honest out-of-sample number. Scaling the
  log-probabilities by ``1/T`` is identical to the textbook logit temperature
  scaling (the per-row ``logsumexp`` constant cancels in the softmax), so it needs
  only ``predict_proba`` and works for binary and multiclass alike.
* **Selective prediction** — risk–coverage curve from the head's confidence; reports
  accuracy@80% coverage (the production "abstain & route to a human" lever).

``python -m asag.models.robustness`` → ``reports/phase4_robust/robustness.json`` + figures.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from asag.config import DataConfig, load_data_config
from asag.features import lexical, negation
from asag.features.build import ENC_COLS, FEAT_COLS, KEY_COLUMNS, _merge_views
from asag.models.data import load_bundle, make_X, make_y
from asag.models.fusion import LIGHTGBM_AVAILABLE, LgbmFusionHead
from asag.models.tasks import REGISTRY, get_spec
from asag.utils.logging import get_logger
from asag.utils.seed import set_global_seed

log = get_logger()

FILLER = " lorem ipsum dolor sit amet consectetur"   # off-topic content lemmas
RNG_SEED = 42


# --------------------------- perturbations -------------------------------

def _perturb(df: pd.DataFrame, kind: str, frac: float = 0.3) -> pd.DataFrame:
    """Return a copy of the merged view with the student feature-view text edited."""
    out = df.copy()
    rng = np.random.default_rng(RNG_SEED)
    s_feat = out["student_answer_feat"].fillna("").astype(str)
    s_neg = out["student_answer_feat_neg"].fillna("").astype(str)

    if kind == "length_pad":
        out["student_answer_feat"] = s_feat + FILLER
        out["student_answer_feat_neg"] = s_neg + FILLER
        out["student_answer_enc"] = out["student_answer_enc"].fillna("").astype(str) + FILLER
    elif kind == "paraphrase_drop":
        def drop(t):
            toks = t.split()
            if len(toks) <= 2:
                return t
            keep = [w for w in toks if rng.random() > frac]
            return " ".join(keep) if keep else toks[0]
        out["student_answer_feat"] = s_feat.map(drop)
        out["student_answer_feat_neg"] = s_neg.map(drop)
    elif kind == "negation_flip":
        # mark every student content token as negated (full-scope negation)
        out["student_answer_feat_neg"] = s_feat.map(
            lambda t: " ".join("neg_" + w for w in t.split()))
    else:
        raise ValueError(kind)
    return out


def _recompute(df: pd.DataFrame, base_feats: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Splice freshly-computed lexical+negation columns over the cached feature row."""
    feats = base_feats.copy()
    lex = lexical.compute_lexical(df, cfg)
    neg = negation.compute_negation(df, cfg)
    for col in lex.columns:
        feats[col] = lex[col].values
    for col in neg.columns:
        feats[col] = neg[col].values
    return feats


def _perturbation_suite(name: str, cfg: DataConfig, n_sample: int = 300) -> dict:
    spec = get_spec(name)
    bundle = load_bundle(name, cfg, spec)
    ddir = cfg.paths.processed / name
    merged = _merge_views(name, pd.read_parquet(ddir / "encoder.parquet"),
                          pd.read_parquet(ddir / "feature.parquet"))
    has_ref = merged["reference_answer_feat"].fillna("").astype(str).str.strip().ne("").any()
    if not has_ref:
        return {"status": "skipped", "reason": "no reference answer (lexical features NaN)"}

    # train one head on all rows, probe a held-out sample (kept out of training)
    y_all = make_y(bundle.df, bundle)
    finite = np.isfinite(y_all)
    idx = np.where(finite & merged["reference_answer_feat"].fillna("").astype(str).str.strip().ne("").values)[0]
    rng = np.random.default_rng(RNG_SEED)
    sample = rng.choice(idx, size=min(n_sample, len(idx)), replace=False)
    train_mask = finite.copy(); train_mask[sample] = False

    set_global_seed(RNG_SEED)
    head = LgbmFusionHead(spec.task_type, cfg.model.lightgbm, RNG_SEED).fit(
        make_X(bundle.df[train_mask], bundle.feature_cols), y_all[train_mask])

    base_row = bundle.df.iloc[sample][bundle.feature_cols].reset_index(drop=True)
    merged_s = merged.iloc[sample].reset_index(drop=True)
    pred0 = head.predict(base_row)

    res = {"status": "ok", "n_sample": int(len(sample)), "task_type": spec.task_type,
           "perturbations": {}}
    for kind in ("length_pad", "paraphrase_drop", "negation_flip"):
        pert = _recompute(_perturb(merged_s, kind), base_row, cfg)
        pred1 = head.predict(pert[bundle.feature_cols])
        delta = pred1 - pred0
        res["perturbations"][kind] = {
            "mean_abs_delta": round(float(np.mean(np.abs(delta))), 4),
            "mean_signed_delta": round(float(np.mean(delta)), 4),
            "frac_changed": round(float(np.mean(np.abs(delta) > 1e-9)), 4),
        }
    return res


# --------------------------- calibration ---------------------------------

def expected_calibration_error(y_true, proba, n_bins: int = 10) -> tuple[float, list]:
    """ECE for the predicted-class confidence (top-1), + per-bin reliability."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, diagram = 0.0, []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if not m.any():
            continue
        acc, avg_conf, w = correct[m].mean(), conf[m].mean(), m.mean()
        ece += w * abs(acc - avg_conf)
        diagram.append({"bin": round(float(bins[i + 1]), 2), "acc": round(float(acc), 4),
                        "conf": round(float(avg_conf), 4), "weight": round(float(w), 4)})
    return round(float(ece), 4), diagram


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def temperature_scale(proba: np.ndarray, temp: float) -> np.ndarray:
    """Apply temperature ``T`` to a probability matrix via its log-probabilities.

    ``softmax(log(p)/T)`` equals the standard ``softmax(z/T)`` on the classifier's
    logits ``z`` (the per-row ``logsumexp`` offset is constant and cancels), so this
    is exact temperature scaling without needing raw margins. ``T>1`` softens
    (reduces) confidence; ``T<1`` sharpens it.
    """
    logp = np.log(np.clip(proba, 1e-12, 1.0))
    return _softmax_rows(logp / temp)


def fit_temperature(proba_val: np.ndarray, y_val: np.ndarray,
                    grid: np.ndarray | None = None) -> float:
    """Pick ``T>0`` minimising validation NLL by deterministic grid search.

    Grid search (no scipy/optimiser) keeps it dependency-free and reproducible.
    """
    if grid is None:
        grid = np.concatenate([np.linspace(0.5, 1.0, 11), np.linspace(1.05, 10.0, 180)])
    y = y_val.astype(int)
    rows = np.arange(len(y))
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        sp = temperature_scale(proba_val, float(t))
        nll = float(-np.mean(np.log(np.clip(sp[rows, y], 1e-12, 1.0))))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def risk_coverage(y_true, proba, points=(1.0, 0.9, 0.8, 0.7, 0.5)) -> list:
    """Accuracy at fixed coverage levels, keeping the most-confident predictions."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    order = np.argsort(-conf)
    out = []
    for cov in points:
        k = max(1, int(cov * len(conf)))
        out.append({"coverage": cov, "accuracy": round(float(correct[order[:k]].mean()), 4)})
    return out


def _aligned_proba(head: LgbmFusionHead, X) -> np.ndarray:
    """``predict_proba`` with columns reordered to sorted class-code order."""
    proba = head.model.predict_proba(X)
    col = {int(c): j for j, c in enumerate(head.model.classes_)}
    return proba[:, [col[c] for c in sorted(col)]]


def _calibration(name: str, cfg: DataConfig) -> dict:
    spec = get_spec(name)
    if spec.task_type != "classification":
        return {"status": "n/a", "reason": "ECE/abstention implemented for classification heads"}
    bundle = load_bundle(name, cfg, spec)
    df = bundle.df
    y = make_y(df, bundle)
    finite = np.isfinite(y)
    # Three-way split: fit head on train, fit temperature on val, score ECE on test.
    if spec.protocol == "official_split":
        tr_all = np.where((df["split"] == "train").to_numpy() & finite)[0]
        te = (df["split"] == spec.test_splits[-1]).to_numpy() & finite
        rng = np.random.default_rng(RNG_SEED)
        tr_all = rng.permutation(tr_all)
        cut = max(1, int(0.2 * len(tr_all)))
        val_idx, tr_idx = tr_all[:cut], tr_all[cut:]
        val = np.zeros(len(df), bool); val[val_idx] = True
        tr = np.zeros(len(df), bool); tr[tr_idx] = True
    else:  # grouped OOF: fold 0 = test, fold 1 = val, folds ≥2 = train (questions disjoint)
        te = (df["fold"] == 0).to_numpy() & finite
        val = (df["fold"] == 1).to_numpy() & finite
        tr = (df["fold"] >= 2).to_numpy() & finite
    if tr.sum() == 0 or te.sum() == 0 or val.sum() == 0:
        return {"status": "degenerate"}

    set_global_seed(RNG_SEED)
    head = LgbmFusionHead("classification", cfg.model.lightgbm, RNG_SEED).fit(
        make_X(df[tr], bundle.feature_cols), y[tr])
    proba_te = _aligned_proba(head, make_X(df[te], bundle.feature_cols))
    proba_val = _aligned_proba(head, make_X(df[val], bundle.feature_cols))
    yte, yval = y[te].astype(int), y[val].astype(int)

    temp = fit_temperature(proba_val, yval)
    proba_cal = temperature_scale(proba_te, temp)
    ece_pre, diag_pre = expected_calibration_error(yte, proba_te)
    ece_post, diag_post = expected_calibration_error(yte, proba_cal)
    return {"status": "ok", "n_test": int(te.sum()), "n_val": int(val.sum()),
            "temperature": round(temp, 4),
            "ece_pre": ece_pre, "ece_post": ece_post,
            "reliability_pre": diag_pre, "reliability_post": diag_post,
            "risk_coverage": risk_coverage(yte, proba_cal)}


def run_all(cfg: DataConfig | None = None, names: list[str] | None = None) -> dict:
    cfg = cfg or load_data_config()
    if not LIGHTGBM_AVAILABLE:
        raise RuntimeError("lightgbm is not installed; run `uv pip install lightgbm`")
    names = names or [n for n in REGISTRY
                      if (cfg.paths.processed / n / "features.parquet").exists()]
    out: dict[str, dict] = {}
    for name in names:
        out[name] = {"perturbations": _perturbation_suite(name, cfg),
                     "calibration": _calibration(name, cfg)}
        p = out[name]["perturbations"]
        if p.get("status") == "ok":
            nf = p["perturbations"]["negation_flip"]["mean_signed_delta"]
            lp = p["perturbations"]["length_pad"]["mean_abs_delta"]
            log.info(f"{name}: negation_flip Δ={nf:+.4f}  length_pad |Δ|={lp:.4f}")
        c = out[name]["calibration"]
        if c.get("status") == "ok":
            log.info(f"{name}: ECE {c['ece_pre']:.4f} → {c['ece_post']:.4f} "
                     f"(T={c['temperature']:.2f})")
    dst = cfg.paths.reports / "phase4_robust"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "robustness.json").write_text(json.dumps({"datasets": out}, indent=2), encoding="utf-8")
    log.info(f"wrote {dst / 'robustness.json'}")
    _write_calibration_figure(cfg, out)
    return {"datasets": out}


def _write_calibration_figure(cfg: DataConfig, out: dict) -> None:
    """Reliability diagrams (pre vs post temperature scaling) per classification head."""
    cal = {n: d["calibration"] for n, d in out.items()
           if d.get("calibration", {}).get("status") == "ok"}
    if not cal:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cal), figsize=(4.2 * len(cal), 4), squeeze=False)
    for ax, (name, c) in zip(axes[0], cal.items()):
        ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfect")
        for diag, style, lab in ((c["reliability_pre"], "o-", f"pre (ECE={c['ece_pre']:.3f})"),
                                 (c["reliability_post"], "s-", f"post (ECE={c['ece_post']:.3f})")):
            xs = [d["conf"] for d in diag]; ys = [d["acc"] for d in diag]
            ax.plot(xs, ys, style, ms=4, label=lab)
        ax.set_title(f"{name}  (T={c['temperature']:.2f})", fontsize=10)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Phase 4 — reliability before/after temperature scaling")
    fig.tight_layout()
    fig.savefig(cfg.paths.figures / "phase4_calibration.png", dpi=120)
    plt.close(fig)
    log.info(f"wrote {cfg.paths.figures / 'phase4_calibration.png'}")


if __name__ == "__main__":
    import sys
    run_all(names=sys.argv[1:] or None)
