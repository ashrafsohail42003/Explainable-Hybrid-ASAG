"""Per-dataset task registry — the single source of truth for Phase 2C.

Each :class:`TaskSpec` declares how a processed dataset is modelled and scored:

* ``task_type``  — ``classification`` | ``ordinal`` | ``regression``. Drives the
  head (classifier vs regressor) and the target column.
* ``target``     — ``label`` (classification) or ``score`` (ordinal/regression).
* ``protocol``   — ``official_split`` (fit on ``train``, score each held-out test
  split) or ``kfold`` (rotate the materialized ``fold`` column built in Phase 1).
* ``per_prompt`` — train/score one model **per ``question_id``** and average the
  headline metric (ASAP-SAS: each prompt has its own rubric/scale — the report
  explicitly warns against pooling prompts into one metric).
* ``metrics``    — which metrics to compute (see :mod:`asag.models.metrics`).
* ``headline``   — the metric reported as the dataset's bar in the summary figure.

The heterogeneity here is deliberate: SemEval is 5-way classification (macro-F1),
ASAP-SAS / MIND-CA are ordinal (QWK), Mohler / SAF are regression (Pearson/RMSE),
Powergrading is binary. One model family (NaN-native GBM) serves all of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    task_type: str            # classification | ordinal | regression
    target: str               # label | score
    protocol: str             # official_split | kfold
    metrics: tuple[str, ...]
    headline: str
    per_prompt: bool = False
    test_splits: tuple[str, ...] = ()   # only for protocol == official_split


REGISTRY: dict[str, TaskSpec] = {
    "semeval": TaskSpec(
        dataset="semeval",
        task_type="classification",
        target="label",
        protocol="official_split",
        metrics=("macro_f1", "weighted_f1", "accuracy"),
        headline="macro_f1",
        test_splits=("test_ua", "test_uq", "test_ud"),   # test_ud = cross-domain headline
    ),
    "saf": TaskSpec(
        dataset="saf",
        task_type="regression",
        target="score",
        protocol="official_split",
        metrics=("rmse", "mae", "pearson", "spearman"),
        headline="pearson",
        test_splits=("test_ua", "test_uq"),
    ),
    "asap_sas": TaskSpec(
        dataset="asap_sas",
        task_type="ordinal",
        target="score",
        protocol="official_split",
        metrics=("qwk", "accuracy"),
        headline="qwk",
        per_prompt=True,                                 # one model per EssaySet (set_<n>)
        test_splits=("test_ua",),
    ),
    "mohler": TaskSpec(
        dataset="mohler",
        task_type="regression",
        target="score",
        protocol="kfold",
        metrics=("pearson", "spearman", "rmse", "mae"),
        headline="pearson",
    ),
    "powergrading": TaskSpec(
        dataset="powergrading",
        task_type="classification",
        target="label",
        protocol="kfold",
        metrics=("macro_f1", "accuracy", "qwk"),
        headline="macro_f1",
    ),
    "mindreading": TaskSpec(
        dataset="mindreading",
        task_type="ordinal",
        target="score",
        protocol="kfold",
        metrics=("qwk", "accuracy"),
        headline="qwk",
    ),
}


def get_spec(name: str) -> TaskSpec:
    if name not in REGISTRY:
        raise KeyError(f"no task spec for dataset {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]
