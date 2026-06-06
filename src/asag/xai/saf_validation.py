"""Phase 2F — validate explanations against SAF human gold feedback (the novelty).

SAF (Filighera 2022) ships, per answer, a human ``verification_feedback`` verdict
(``Correct`` / ``Partially correct`` / ``Incorrect``) and a natural-language
``answer_feedback`` rationale — dropped by the unified schema, recovered here from
the raw parquets.

The claim we test quantitatively: *the model's interpretable rubric-coverage
signal tracks the human verdict*. For each interpretable signal we recompute it
directly from the raw (reference, answer) text (so no row alignment is needed),
then measure how well it separates the three verdict classes:

* mean signal per class (should rise Incorrect → Partially correct → Correct),
* Spearman ρ between the signal and the ordinal verdict (monotonic alignment),
* ROC-AUC of the signal predicting ``Correct`` vs the rest.

This is the report's "attributions vs gold feedback" move: it shows the
explanations are not decorative but agree with how humans graded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from asag.config import DataConfig, load_data_config
from asag.xai.concept_attribution import concept_coverage

# raw SAF split files that carry the gold feedback, mapped to fair held-out labels.
_HELD_OUT = {"test_unseen_answers.parquet": "test_ua",
             "test_unseen_questions.parquet": "test_uq"}
_VERDICT_ORDER = {"Incorrect": 0, "Partially correct": 1, "Correct": 2}


def _signals(refs: list[str], answers: list[str], encoder, nlp, tau: float) -> pd.DataFrame:
    """Recompute the interpretable coverage signals from raw text."""
    cov = concept_coverage(refs, answers, encoder, nlp, tau)
    rub_cov, rub_mean, rub_min = [], [], []
    for concepts in cov:
        if not concepts:
            rub_cov.append(np.nan); rub_mean.append(np.nan); rub_min.append(np.nan)
            continue
        sims = np.array([c["similarity"] for c in concepts])
        rub_cov.append(float((sims >= tau).mean()))
        rub_mean.append(float(sims.mean()))
        rub_min.append(float(sims.min()))
    u = encoder.embed([a or "" for a in answers])
    v = encoder.embed([r or "" for r in refs])
    sem = np.einsum("ij,ij->i", u, v).astype(float)
    return pd.DataFrame({
        "rub_coverage_at_tau": rub_cov, "rub_mean_maxsim": rub_mean,
        "rub_min_maxsim": rub_min, "sem_cosine": sem,
    })


def _signal_stats(sig: np.ndarray, ordinal: np.ndarray, verdict: np.ndarray) -> dict:
    m = np.isfinite(sig)
    s, o, v = sig[m], ordinal[m], verdict[m]
    by_class = {name: round(float(np.mean(s[v == name])), 4)
                for name in _VERDICT_ORDER if (v == name).any()}
    rho = spearmanr(s, o).correlation if np.ptp(o) > 0 and s.size > 2 else float("nan")
    correct = (v == "Correct").astype(int)
    auc = (roc_auc_score(correct, s)
           if correct.min() != correct.max() and s.size > 2 else float("nan"))
    return {
        "by_class": by_class,
        "spearman_vs_verdict": None if not np.isfinite(rho) else round(float(rho), 3),
        "auc_correct_vs_rest": None if not np.isfinite(auc) else round(float(auc), 3),
        "monotonic": bool(len(by_class) == 3 and
                          by_class["Incorrect"] <= by_class["Partially correct"] <= by_class["Correct"]),
    }


def validate_saf(cfg: DataConfig | None = None, encoder=None, nlp=None) -> dict:
    """Test whether SAF coverage signals align with the human gold verdicts."""
    cfg = cfg or load_data_config()
    ds = cfg.datasets["saf"]
    saf_dir = cfg.paths.raw / ds.raw_subdir
    if not saf_dir.exists():
        return {"status": "missing", "reason": f"no raw SAF under {saf_dir}"}

    frames = []
    for fname, split in _HELD_OUT.items():
        fp = saf_dir / fname
        if fp.exists():
            d = pd.read_parquet(fp)
            d = d[["reference_answer", "provided_answer", "verification_feedback"]].copy()
            d["split"] = split
            frames.append(d)
    if not frames:
        return {"status": "missing", "reason": "no SAF held-out parquets with feedback"}
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["verification_feedback"].isin(_VERDICT_ORDER)]

    if encoder is None:
        from asag.features.semantic import SbertEncoder
        encoder = SbertEncoder(cfg.features.sbert_model,
                               batch_size=cfg.features.semantic.batch_size, normalize=True)
    if nlp is None:
        from asag.features.text_utils import load_feature_nlp
        nlp = load_feature_nlp(cfg.features.ner.spacy_model)

    sig = _signals(raw["reference_answer"].tolist(), raw["provided_answer"].tolist(),
                   encoder, nlp, cfg.features.rubric.tau)
    verdict = raw["verification_feedback"].to_numpy()
    ordinal = raw["verification_feedback"].map(_VERDICT_ORDER).to_numpy(dtype=float)

    signals = {col: _signal_stats(sig[col].to_numpy(), ordinal, verdict) for col in sig.columns}
    return {
        "status": "ok",
        "n": int(len(raw)),
        "splits": sorted(raw["split"].unique().tolist()),
        "tau": cfg.features.rubric.tau,
        "verification_counts": {k: int(v) for k, v in raw["verification_feedback"].value_counts().items()},
        "signals": signals,
    }
