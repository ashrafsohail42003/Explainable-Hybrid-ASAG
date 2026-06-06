"""Phase 2D — inter-annotator agreement (IAA), the human ceiling.

Reviewers expect a human-agreement reference: the QWK a model is measured against
is only meaningful next to the QWK *two human raters* reach on the same rubric.
ASAP-SAS is the one dataset in our stack that ships two independent grader reads
(``Score1`` / ``Score2``) — the unified loader keeps only ``Score1`` as the
target, so we recover ``Score2`` straight from the raw TSVs here.

Important data quirk (verified on the AERA mirror): ``Score2`` is fully populated
on **train** and **dev** but **withheld on the test split**. The IAA is therefore
measured on the train+dev rater reads — it is a corpus/rubric property, reported
as the ceiling next to the model's test QWK, not a per-test-item agreement.

Per ``EssaySet`` (= ``question_id`` ``set_<n>``) we compute QWK(Score1, Score2)
and macro-average across prompts, mirroring the per-prompt ASAP-SAS protocol.

Other datasets have no second-grader column in our schema (Mohler already averages
its two raters; SAF/MIND-CA expose a single gold score), so their ceiling is
reported as ``unavailable`` with a one-line reason — a documented gap, not silent.
"""

from __future__ import annotations

import pandas as pd

from asag.config import DataConfig, load_data_config
from asag.models.metrics import qwk
from asag.utils.logging import get_logger

log = get_logger()

_ASAP_SPLIT_FILES = ("train.tsv", "dev.tsv", "test.tsv")

# Why every non-ASAP dataset lacks a reconstructable human ceiling in our schema.
_NO_CEILING_REASON = {
    "mohler": "score is the average of two graders; per-grader reads not retained",
    "saf": "single gold score per answer; no second-grader column",
    "mindreading": "single ordinal annotation per response in our schema",
    "semeval": "5-way entailment labels; no second-annotator column retained",
    "powergrading": "binary consensus label; per-grader reads not retained",
}


def asap_sas_ceiling(cfg: DataConfig | None = None) -> dict:
    """QWK between the two ASAP-SAS grader reads, per prompt and macro-averaged."""
    cfg = cfg or load_data_config()
    ds = cfg.datasets["asap_sas"]
    asap_dir = cfg.paths.raw / ds.raw_subdir
    if not asap_dir.exists():
        return {"status": "missing", "reason": f"no raw ASAP-SAS under {asap_dir}"}

    frames = []
    splits_used = []
    for fname in _ASAP_SPLIT_FILES:
        fpath = asap_dir / fname
        if not fpath.exists():
            continue
        raw = pd.read_csv(fpath, sep="\t")
        s1 = pd.to_numeric(raw.get("Score1"), errors="coerce")
        s2 = pd.to_numeric(raw.get("Score2"), errors="coerce")
        block = pd.DataFrame({"set": raw["EssaySet"].astype(str), "s1": s1, "s2": s2})
        block = block[block["s1"].notna() & block["s2"].notna()]
        if not block.empty:
            frames.append(block)
            splits_used.append(fname.replace(".tsv", ""))

    if not frames:
        return {"status": "no_second_read",
                "reason": "Score2 not populated in any split (test split withholds it)"}

    both = pd.concat(frames, ignore_index=True)
    per_prompt: dict[str, dict] = {}
    for s, g in both.groupby("set"):
        per_prompt[f"set_{s}"] = {"qwk": round(qwk(g["s1"], g["s2"]), 4), "n": int(len(g))}
    macro = float(pd.Series([v["qwk"] for v in per_prompt.values()]).mean())

    log.info(f"asap_sas IAA ceiling: macro QWK={macro:.4f} over {len(per_prompt)} prompts "
             f"(splits: {'+'.join(splits_used)}; Score2 withheld on test)")
    return {
        "status": "ok",
        "metric": "qwk",
        "splits_used": splits_used,
        "note": "Score2 is withheld on the test split; IAA measured on train+dev rater reads.",
        "per_prompt": per_prompt,
        "macro_qwk": round(macro, 4),
        "n_pairs": int(len(both)),
    }


def ceiling_for(name: str, cfg: DataConfig | None = None) -> dict:
    """Human-ceiling (IAA) for a dataset; ``unavailable`` with a reason elsewhere."""
    if name == "asap_sas":
        return asap_sas_ceiling(cfg)
    return {"status": "unavailable",
            "reason": _NO_CEILING_REASON.get(name, "no second-grader column in our schema")}
