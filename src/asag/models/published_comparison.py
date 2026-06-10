"""Phase 3 — head-to-head against published numbers (with honest comparability).

A reviewer's first question is "how does this compare to prior work on the same
data?". This module assembles that table — but its more important job is to state,
per dataset, **whether a comparison is even valid**, because two of our corpora are
*subsets* of the originals and a naive "we beat / we trail" claim on them would be
misleading:

* **Mohler** — we use the ASAG2024 mirror (21 questions / ~1.3k rows), not the full
  Mohler-2011 corpus (80 questions / 2.3k). Pearson is **not comparable**.
* **ASAP-SAS** — the free AERA mirror covers EssaySets 1/2/5/6 (4 of 10 prompts).
  Our macro-QWK is over those 4; the Kaggle leaderboard is over all 10. **Partly
  comparable** (same prompts, different prompt set) — never quote the LB number.
* **SemEval-2013 Task 7** — official splits, full SciEntsBank+Beetle 5-way. **Directly
  comparable** to the shared-task and later transformer papers.
* **SAF / Powergrading / MIND-CA** — niche; published baselines differ in metric or
  setup, flagged individually.

Each published anchor carries a citation and ``needs_verification: true`` — the
numbers here are the author's best recollection of the literature and **must be
checked against the cited paper before the camera-ready**. The module never invents
a verdict; it prints ours next to theirs and the comparability flag, and leaves the
claim to the author.

    python -m asag.models.published_comparison   # -> reports/phase3/published_comparison.{json,md}
"""

from __future__ import annotations

import json

from asag.config import DataConfig, ensure_dirs, load_data_config
from asag.utils.logging import get_logger

log = get_logger()

# Comparability + published anchors. metric matches our headline where possible.
# NOTE: every `published` value is flagged needs_verification — verify before submit.
# ``claim_allowed`` = may we write a "we match / trail X" sentence at all? Only
# when the metric, data, AND test split line up. Everything else is reported as an
# *internal* number for context, never as a head-to-head. ``split_match`` records
# the remaining caveat even where a claim is allowed.
REFERENCE = {
    "semeval": {
        "comparability": "context-only — split must be matched (we headline the hard test_ud)",
        "claim_allowed": False,
        "metric": "macro_f1",
        "split_note": ("SciEntsBank+Beetle 5-way. Our headline is the unseen-domain "
                       "test_ud (the hardest split); most published 5-way macro-F1 are "
                       "on unseen-answers (test_ua). A claim is only valid split-matched — "
                       "report our test_ua next to test_ua numbers, not test_ud."),
        "published": [
            {"system": "SemEval-2013 shared task, best 5-way macro-F1 (SciEntsBank, test_ua)",
             "value": "~0.55-0.62", "cite": "Dzikovska et al. 2013, S13-2045"},
            {"system": "BERT fine-tuned, SciEntsBank 5-way (verify split + exact value)",
             "value": "~0.58", "cite": "Sung et al. 2019"},
        ],
    },
    "asap_sas": {
        "comparability": "not comparable — 4/10 prompts (AERA mirror) vs all-10 published",
        "claim_allowed": False,
        "metric": "qwk",
        "split_note": ("EssaySets 1/2/5/6 only (the free AERA mirror). Published QWK is "
                       "Fisher-averaged over all 10 prompts, so it is NOT our prompt set; "
                       "the Kaggle LB (~0.78, all 10) is likewise off-limits."),
        "published": [
            {"system": "Neural LSTM/attention, ASAP-SAS mean QWK (Fisher-avg, all 10 prompts)",
             "value": "~0.74", "cite": "Riordan, Horbach et al. 2017, BEA W17-5017"},
            {"system": "Kaggle ASAP-SAS private LB top (all 10 prompts)",
             "value": "~0.78", "cite": "Kaggle 2012 (not our prompt set)"},
        ],
    },
    "mohler": {
        "comparability": "not comparable — ASAG2024 subset (21q) vs full corpus (80q)",
        "claim_allowed": False,
        "metric": "pearson",
        "split_note": "report as an internal number only; do not claim vs Mohler-2011",
        "published": [
            {"system": "Mohler et al. 2011 best (full 80-question corpus)",
             "value": "r~0.52", "cite": "Mohler et al. 2011"},
        ],
    },
    "saf": {
        "comparability": "metric-mismatch — published reports RMSE/feedback-F1",
        "claim_allowed": False,
        "metric": "pearson",
        "split_note": ("we report Pearson; SAF paper centers RMSE + verification-feedback F1. "
                       "We treat SAF as the explainability case study (it uniquely ships gold "
                       "feedback), not an accuracy headline — our test_uq Pearson is ~0 and "
                       "the gain is NOT significant (see significance.json)."),
        "published": [
            {"system": "SAF baseline (T5 / encoder), score RMSE",
             "value": "RMSE-based", "cite": "Filighera et al. 2022"},
        ],
    },
    "powergrading": {
        "comparability": "setup-mismatch — original is clustering, not supervised F1",
        "claim_allowed": False,
        "metric": "macro_f1",
        "split_note": ("Basu 2013 frames it as answer clustering; our supervised F1 is a "
                       "different task. NB: under the cluster (question-level) bootstrap the "
                       "head's gain over baseline is NOT significant (only 20 questions)."),
        "published": [
            {"system": "Powergrading clustering (Basu et al. 2013)",
             "value": "clustering metrics", "cite": "Basu et al. 2013"},
        ],
    },
    "mindreading": {
        "comparability": "approximate — same corpus, check exact metric/setup",
        "claim_allowed": False,
        "metric": "qwk",
        "split_note": "Kovatchev 2020 reports accuracy/F1 on the 0/1/2 task; we report QWK",
        "published": [
            {"system": "MIND-CA baselines (Kovatchev et al. 2020)",
             "value": "accuracy/F1 reported", "cite": "Kovatchev et al. 2020"},
        ],
    },
}


def _our_numbers(cfg: DataConfig) -> dict[str, dict]:
    """Headline mean for each dataset from 2C (default), 2D (tuned), 2G (neural)."""
    out: dict[str, dict] = {}

    def headline(doc_path, model_key):
        if not doc_path.exists():
            return {}
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        res = {}
        for n, r in doc.get("datasets", {}).items():
            h = r.get("headline", {})
            res[n] = h.get(model_key, {}).get("mean")
        return res

    def neural_only(doc_path):
        """Raw DeBERTa headline from the Phase B three-way report (preferred)."""
        if not doc_path.exists():
            return {}
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        return {n: r.get("neural_only", {}).get("mean")
                for n, r in doc.get("datasets", {}).items() if r.get("status") == "ok"}

    g2c = headline(cfg.paths.reports / "phase2c" / "results.json", "gbm")
    g2d = headline(cfg.paths.reports / "phase2d" / "results.json", "gbm")
    # neural-only: prefer the Phase B three-way report, fall back to the phase2g eval
    g2g = {**headline(cfg.paths.reports / "phase2g" / "results.json", "neural"),
           **neural_only(cfg.paths.reports / "phase_hybrid" / "three_way.json")}
    for n in REFERENCE:
        out[n] = {"gbm_2c": g2c.get(n), "gbm_2d_tuned": g2d.get(n), "neural_2g": g2g.get(n)}
    return out


def build_comparison(cfg: DataConfig | None = None) -> dict:
    cfg = cfg or load_data_config()
    ensure_dirs(cfg)
    ours = _our_numbers(cfg)
    table = {}
    for n, ref in REFERENCE.items():
        table[n] = {"metric": ref["metric"], "comparability": ref["comparability"],
                    "claim_allowed": ref.get("claim_allowed", False),
                    "split_note": ref["split_note"], "ours": ours.get(n, {}),
                    "published": [{**p, "needs_verification": True} for p in ref["published"]]}
    out_dir = cfg.paths.reports / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "published_comparison.json").write_text(
        json.dumps({"datasets": table,
                    "disclaimer": "Published values are recollections; verify each against its "
                                  "citation before submission. Comparability flags state where a "
                                  "direct claim is invalid (subset/metric/setup mismatch)."},
                   indent=2, default=str), encoding="utf-8")
    _write_md(out_dir / "published_comparison.md", table)
    log.info(f"wrote {out_dir}/published_comparison.json, .md")
    return table


def _fmt(v) -> str:
    return "—" if v is None else (f"{v:.4f}" if isinstance(v, (int, float)) else str(v))


def _write_md(path, table: dict) -> None:
    lines = ["# Phase 3 — Head-to-head vs published (verify before submission)\n",
             "> Published numbers are author recollections flagged `needs_verification`.",
             "> **Claim?** = is a direct \"we match/trail X\" sentence valid (metric + data +",
             "> split all aligned)? Currently **no dataset** clears that bar — every row is",
             "> reported for *context only*. **Comparability** states why.\n",
             "| Dataset | Metric | Ours (2C / 2D / neural) | Published (cite) | Claim? | Comparability |",
             "|---|---|---|---|---|---|"]
    for n, t in table.items():
        o = t["ours"]
        ours = f"{_fmt(o.get('gbm_2c'))} / {_fmt(o.get('gbm_2d_tuned'))} / {_fmt(o.get('neural_2g'))}"
        pub = "; ".join(f"{p['system'].split('(')[0].strip()} {p['value']} [{p['cite']}]"
                        for p in t["published"])
        claim = "✅ yes" if t.get("claim_allowed") else "❌ context-only"
        lines.append(f"| {n} | {t['metric']} | {ours} | {pub} | {claim} | {t['comparability']} |")
    lines += ["\n## Per-dataset notes\n"]
    for n, t in table.items():
        lines.append(f"- **{n}** — {t['split_note']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_comparison()
