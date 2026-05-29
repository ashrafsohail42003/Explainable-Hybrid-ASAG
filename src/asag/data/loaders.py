"""Per-dataset loaders. Each returns a DataFrame in the unified schema.

Unified schema (one row per (question, student_answer)):

    question_id: str
    question: str
    reference_answer: str
    student_answer: str
    score: float        # raw original score; NEVER normalized destructively
    label: str          # 5-way categorical when available; else "" (empty string)
    dataset: str
    domain: str
    split: str          # train|dev|test_ua|test_uq|test_ud|prompt_specific_test|all

Datasets without an official label scheme (e.g. Mohler) leave ``label = ""``.
Datasets without official splits (e.g. Mohler) emit ``split = "all"`` and rely
on :mod:`asag.data.splits` to build stratified k-fold CV indices.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import pandas as pd

from asag.config import DataConfig, load_data_config
from asag.utils.logging import get_logger

log = get_logger()


UNIFIED_COLUMNS = [
    "question_id",
    "question",
    "reference_answer",
    "student_answer",
    "score",
    "label",
    "dataset",
    "domain",
    "split",
]


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    for col in ("question_id", "question", "reference_answer", "student_answer",
                "label", "dataset", "domain", "split"):
        df[col] = df[col].astype("string").fillna("")
    return df[UNIFIED_COLUMNS]


# ---------------- SemEval-2013 Task 7 ----------------

_SEMEVAL_DOMAIN = {"beetle": "electronics", "sciEntsBank": "science"}
_SEMEVAL_SPLIT_DIRS = {
    "train": "train",
    "training": "train",  # legacy variant
    "test-unseen-answers": "test_ua",
    "test-unseen-questions": "test_uq",
    "test-unseen-domains": "test_ud",
}


def _iter_semeval_files(root: Path) -> Iterable[tuple[str, str, Path]]:
    """Yield (corpus, split_label, xml_path) tuples for the 5-way semeval data.

    Each split directory contains Core/, Extra/, and (sometimes) Dependency/
    subdirectories. These are alternative annotation styles over the SAME
    student responses; reading all of them produces 2-3x duplicated rows.
    Standard practice in the ASAG literature is to use Core as the
    canonical primary annotation, which is what we do here.

    The 5-way zip layout::

        semeval-5way/
          beetle/{train, test-unseen-answers, test-unseen-questions}/{Core, Extra, Dependency}/*.xml
          sciEntsBank/{train, test-unseen-answers, test-unseen-questions, test-unseen-domains}/{Core, Extra, Dependency}/*.xml
          sciEntsBank/reliability/round1/...  (IAA subset; skipped)
    """
    for corpus_dir in root.glob("*/"):
        if not corpus_dir.is_dir():
            continue
        corpus_name = corpus_dir.name  # 'beetle' or 'sciEntsBank'
        if corpus_name not in _SEMEVAL_DOMAIN:
            continue
        for split_name, split_label in _SEMEVAL_SPLIT_DIRS.items():
            sd = corpus_dir / split_name
            if not sd.exists():
                continue
            # Prefer Core/; fall back to the split dir root if no Core/ exists.
            core_dir = sd / "Core"
            scan_root = core_dir if core_dir.exists() else sd
            for xml_path in scan_root.glob("*.xml"):
                yield corpus_name, split_label, xml_path


def _parse_semeval_xml(xml_path: Path) -> list[dict]:
    """Parse a single SemEval question XML into row dicts (5-way labels)."""
    rows: list[dict] = []
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        log.warning(f"semeval: bad XML {xml_path}: {e}")
        return rows
    root = tree.getroot()
    q_id = root.attrib.get("id", xml_path.stem)
    q_text_el = root.find("questionText")
    q_text = (q_text_el.text or "").strip() if q_text_el is not None else ""

    # reference answers: pick best/canonical when multiple exist
    ref_answers = root.find("referenceAnswers")
    ref_text = ""
    if ref_answers is not None:
        candidates = list(ref_answers.findall("referenceAnswer"))
        # Prefer category="BEST" or "GOOD"; else first
        canonical = next(
            (r for r in candidates if (r.attrib.get("category") or "").upper() == "BEST"),
            None,
        ) or (candidates[0] if candidates else None)
        if canonical is not None:
            ref_text = (canonical.text or "").strip()

    student_answers = root.find("studentAnswers")
    if student_answers is None:
        return rows
    for sa in student_answers.findall("studentAnswer"):
        label = (sa.attrib.get("accuracy") or "").strip()
        rows.append({
            "question_id": q_id,
            "question": q_text,
            "reference_answer": ref_text,
            "student_answer": (sa.text or "").strip(),
            "score": float("nan"),  # SemEval 5-way is categorical
            "label": label,
            "domain": "",  # filled by caller
            "split": "",   # filled by caller
        })
    return rows


def load_semeval(cfg: DataConfig | None = None, corpus: str | None = None) -> pd.DataFrame:
    """Load SemEval-2013 Task 7 (5-way). Optionally filter by corpus ('beetle' or 'sciEntsBank')."""
    cfg = cfg or load_data_config()
    raw_dir = cfg.paths.raw / cfg.datasets["semeval"].raw_subdir / "semeval-5way" / "semeval-5way"
    if not raw_dir.exists():
        # repos sometimes extract one level shallower
        alt = cfg.paths.raw / cfg.datasets["semeval"].raw_subdir / "semeval-5way"
        if alt.exists():
            raw_dir = alt
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"SemEval extracted dir not found under {raw_dir.parent}. Run `make download` first."
        )

    rows: list[dict] = []
    for corpus_name, split_label, xml_path in _iter_semeval_files(raw_dir):
        if corpus and corpus_name != corpus:
            continue
        domain = _SEMEVAL_DOMAIN[corpus_name]
        for r in _parse_semeval_xml(xml_path):
            r["dataset"] = f"semeval_{corpus_name}"
            r["domain"] = domain
            r["split"] = split_label
            rows.append(r)

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("semeval: no rows parsed — check the extraction directory layout")
    return _coerce(df)


# ---------------- SAF Communication Networks English ----------------

_SAF_SPLIT_MAP = {
    "train": "train",
    "validation": "dev",
    "test_unseen_answers": "test_ua",
    "test_unseen_questions": "test_uq",
}


def load_saf(cfg: DataConfig | None = None) -> pd.DataFrame:
    """Load SAF Communication Networks English from the cached parquets in data/raw."""
    cfg = cfg or load_data_config()
    saf_dir = cfg.paths.raw / cfg.datasets["saf"].raw_subdir
    if not saf_dir.exists():
        raise FileNotFoundError(f"SAF dir missing: {saf_dir}. Run `make download` first.")

    frames: list[pd.DataFrame] = []
    for parquet_path in sorted(saf_dir.glob("*.parquet")):
        split_name = parquet_path.stem  # e.g. 'train', 'validation', 'test_unseen_answers'
        if split_name not in _SAF_SPLIT_MAP:
            continue
        df = pd.read_parquet(parquet_path)
        # `id` is a per-row UUID; `question_id` semantically must identify
        # the QUESTION, so we hash the question text (stable across rows).
        qid = df["question"].astype(str).map(
            lambda s: hashlib.md5(s.strip().encode("utf-8")).hexdigest()[:16]
        )
        out = pd.DataFrame({
            "question_id": qid,
            "question": df["question"].astype(str),
            "reference_answer": df["reference_answer"].astype(str),
            "student_answer": df["provided_answer"].astype(str),
            "score": pd.to_numeric(df["score"], errors="coerce"),
            "label": df.get("verification_feedback", pd.Series([""] * len(df))).astype(str),
            "dataset": "saf_comm_nets",
            "domain": "comm_networks",
            "split": _SAF_SPLIT_MAP[split_name],
        })
        frames.append(out)

    if not frames:
        raise RuntimeError(f"SAF: no parquet splits found under {saf_dir}")
    return _coerce(pd.concat(frames, ignore_index=True))


# ---------------- Mohler 2011 ----------------

def load_mohler(cfg: DataConfig | None = None) -> pd.DataFrame:
    """Load Mohler 2011.

    Primary source: canonical Mohler extracted from ASAG2024 (the unified
    benchmark; subset where ``data_source == "mohler"``) — written by the
    downloader to ``mohler_canonical_from_asag2024.parquet``. This is the
    real Mohler dataset.

    Note: the Kaggle mirror ``mubeenfurqanahmed/automatic-short-answer-
    grading-dataset`` is NOT actual Mohler 2011 — it contains questions
    about plant respiration / meridians, not CS data structures. We do
    not load it. See reports/DATASETS.md for the writeup.
    """
    cfg = cfg or load_data_config()
    moh_dir = cfg.paths.raw / cfg.datasets["mohler"].raw_subdir
    if not moh_dir.exists():
        raise FileNotFoundError(f"Mohler dir missing: {moh_dir}. Run `make download` first.")

    canonical = moh_dir / "mohler_canonical_from_asag2024.parquet"
    if not canonical.exists():
        raise FileNotFoundError(
            f"Canonical Mohler parquet missing: {canonical}. Re-run `make download` "
            "after fixing ASAG2024 access — the canonical subset is extracted from there."
        )
    log.info(f"mohler: loading canonical {canonical.name}")
    df = pd.read_parquet(canonical)

    # ASAG2024 schema: question, provided_answer, reference_answer, grade,
    # normalized_grade, data_source, index, weight. Use hashed question
    # text as question_id (the source loses the original Mohler qid).
    qid = df["question"].astype(str).map(
        lambda s: hashlib.md5(s.strip().encode("utf-8")).hexdigest()[:16]
    )
    out = pd.DataFrame({
        "question_id": qid,
        "question": df["question"].astype(str),
        "reference_answer": df["reference_answer"].astype(str),
        "student_answer": df["provided_answer"].astype(str),
        "score": pd.to_numeric(df["grade"], errors="coerce"),
        "label": "",
        "dataset": "mohler",
        "domain": "cs_data_structures",
        "split": "all",
    })
    return _coerce(out)


# ---------------- ASAP-SAS (optional) ----------------

def load_asap_sas(cfg: DataConfig | None = None) -> pd.DataFrame:
    """Load ASAP-SAS train.tsv. Each prompt (EssaySet) is a separate logical sub-dataset."""
    cfg = cfg or load_data_config()
    asap_dir = cfg.paths.raw / cfg.datasets["asap_sas"].raw_subdir
    if not asap_dir.exists() or not cfg.datasets["asap_sas"].enabled:
        log.info("asap_sas not present or disabled — returning empty DataFrame.")
        return _coerce(pd.DataFrame())

    train_tsvs = list(asap_dir.rglob("train.tsv")) + list(asap_dir.rglob("train_rel_2.tsv"))
    if not train_tsvs:
        raise FileNotFoundError(f"ASAP-SAS: no train tsv under {asap_dir}.")
    train = pd.read_csv(train_tsvs[0], sep="\t")
    out = pd.DataFrame({
        "question_id": train["EssaySet"].astype(str),
        "question": "",                # ASAP-SAS questions live in a separate prompt PDF; left blank for Phase 1
        "reference_answer": "",        # ditto rubric
        "student_answer": train["EssayText"].astype(str),
        "score": pd.to_numeric(train.get("Score1", train.iloc[:, 2]), errors="coerce"),
        "label": "",
        "dataset": "asap_sas_" + train["EssaySet"].astype(str),
        "domain": "mixed",
        "split": "train",
    })
    return _coerce(out)


# ---------------- entrypoint helper ----------------

def load_all(cfg: DataConfig | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_data_config()
    out: dict[str, pd.DataFrame] = {}
    if cfg.datasets["semeval"].enabled:
        out["semeval"] = load_semeval(cfg)
    if cfg.datasets["saf"].enabled:
        out["saf"] = load_saf(cfg)
    if cfg.datasets["mohler"].enabled:
        out["mohler"] = load_mohler(cfg)
    if cfg.datasets["asap_sas"].enabled:
        out["asap_sas"] = load_asap_sas(cfg)
    return out
