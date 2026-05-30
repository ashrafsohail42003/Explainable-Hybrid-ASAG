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

def load_mindreading(cfg: DataConfig | None = None) -> pd.DataFrame:
    """Load MIND-CA (Kovatchev 2020): 11,311 child responses, ordinal 0/1/2 scoring.

    Each xlsx file is one task with columns: Child_ID, Answer, Score, Age, Gender,
    Question. Scoring is ordinal 0/1/2 (poor / partial / full mindreading
    response). Reference answers are NOT provided — these are open-ended
    psychology assessment items where rubrics live in copyrighted test
    materials. We leave ``reference_answer`` empty; the dataset trains the
    ordinal-regression head and the semantic encoder branch without
    benefiting from the rubric-coverage branch (useful as ablation).
    """
    cfg = cfg or load_data_config()
    ds = cfg.datasets.get("mindreading")
    if ds is None or not ds.enabled:
        log.info("mindreading disabled — returning empty DataFrame.")
        return _coerce(pd.DataFrame())

    root = cfg.paths.raw / ds.raw_subdir
    if not root.exists():
        raise FileNotFoundError(f"MindReading dir missing: {root}. Run `make download`.")

    files = sorted(root.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"MindReading: no xlsx under {root}.")

    frames: list[pd.DataFrame] = []
    for xlsx in files:
        df = pd.read_excel(xlsx)
        if not {"Child_ID", "Answer", "Score", "Question"}.issubset(df.columns):
            log.warning(f"mindreading: unexpected columns in {xlsx.name}: {list(df.columns)}")
            continue
        task_id = xlsx.stem  # e.g. "SS_Brian_Text" / "SFQuestion_1_Text"
        # Human-readable question prompt derived from the task id.
        # The actual psychology test question text is copyrighted and
        # not redistributed in the corpus — we use the task family +
        # name as the prompt.
        if task_id.startswith("SS_"):
            short = task_id.replace("SS_", "").replace("_Text", "")
            prompt = f"Strange Stories ({short}): explain the character's behaviour."
        elif task_id.startswith("SFQuestion_"):
            n = task_id.replace("SFQuestion_", "").replace("_Text", "")
            prompt = f"Silent Films Q{n}: explain what happened in the scene."
        else:
            prompt = task_id

        out = pd.DataFrame({
            "question_id": task_id,
            "question": prompt,
            "reference_answer": "",
            "student_answer": df["Answer"].astype(str),
            "score": pd.to_numeric(df["Score"], errors="coerce"),
            "label": df["Score"].map(lambda s: f"score_{int(s)}" if pd.notna(s) else ""),
            "dataset": "mindreading",
            "domain": "mindreading_behavioral",
            "split": "all",
        })
        frames.append(out)

    return _coerce(pd.concat(frames, ignore_index=True))


def load_powergrading(cfg: DataConfig | None = None) -> pd.DataFrame:
    """Load Powergrading 1.0 — 20 US-civics questions, ~13,960 graded student answers.

    Three graders per response (G1, G2, G3) with values in {-1, 0, 1}. We use
    the 698-student file (which is fully graded; the 100-student file is
    ungraded → ``G1=G2=G3=-1``). The continuous ``score`` column is the
    mean grade across the three graders; the ``label`` column records
    the majority vote as ``correct`` / ``incorrect``.
    """
    cfg = cfg or load_data_config()
    ds = cfg.datasets.get("powergrading")
    if ds is None or not ds.enabled:
        log.info("powergrading disabled — returning empty DataFrame.")
        return _coerce(pd.DataFrame())

    root = cfg.paths.raw / ds.raw_subdir
    if not root.exists():
        raise FileNotFoundError(f"Powergrading dir missing: {root}. Run `make download`.")

    # Build question_id -> primary reference answer map. The TSV is ragged
    # (questions with several alternate answers); we take the first answer
    # as the canonical reference.
    qa_path = root / "questions_answer_key.tsv"
    qrows: dict[str, tuple[str, str, str]] = {}
    for i, raw_line in enumerate(qa_path.read_text(encoding="utf-8").splitlines()):
        if i == 0 or not raw_line.strip():
            continue
        fields = raw_line.split("\t")
        qid, question = fields[0].strip(), fields[1].strip()
        alt_answers = [a.strip() for a in fields[2:] if a.strip()]
        primary = alt_answers[0] if alt_answers else ""
        qrows[qid] = (question, primary, " | ".join(alt_answers))

    sa_path = root / "studentanswers_grades_698.tsv"
    sa = pd.read_csv(sa_path, sep="\t", dtype=str)
    # Drop rows with any missing grade (defensive)
    for g in ("G1", "G2", "G3"):
        sa[g] = pd.to_numeric(sa[g], errors="coerce")
    sa = sa.dropna(subset=["G1", "G2", "G3"]).copy()
    # Continuous score = mean of three graders, mapped from {-1,0,1} -> {0, 0.5, 1}
    mean_grade = (sa[["G1", "G2", "G3"]].mean(axis=1) + 1.0) / 2.0
    # Majority-vote label: 1 if majority positive, 0 if majority negative
    majority = (sa[["G1", "G2", "G3"]] > 0).sum(axis=1) >= 2

    out = pd.DataFrame({
        "question_id": sa["Q#"].astype(str),
        "question": sa["Q#"].map(lambda q: qrows.get(q, ("", "", ""))[0]),
        "reference_answer": sa["Q#"].map(lambda q: qrows.get(q, ("", "", ""))[1]),
        "student_answer": sa["answer"].astype(str),
        "score": mean_grade.astype(float),
        "label": majority.map({True: "correct", False: "incorrect"}),
        "dataset": "powergrading",
        "domain": "civics",
        "split": "all",
    })
    return _coerce(out)


# ASAP-SAS prompt -> sub-domain. The AERA mirror ships EssaySets 1, 2, 5, 6.
_ASAP_DOMAIN = {"1": "science", "2": "science", "5": "biology", "6": "biology"}
# raw split file -> unified split label. The Kaggle test labels were withheld
# originally; the mirror restores gold test scores, mapped to test_ua (the same
# prompts recur in train, i.e. unseen *answers*, not unseen questions).
_ASAP_SPLIT_FILES = {"train.tsv": "train", "dev.tsv": "dev", "test.tsv": "test_ua"}


def load_asap_sas(cfg: DataConfig | None = None) -> pd.DataFrame:
    """Load the ASAP-SAS science/biology subset from the AERA mirror.

    Three split files (``train.tsv``/``dev.tsv``/``test.tsv``) carry the ASAP
    columns ``EssaySet, EssayText, Score1, Score2``. Each EssaySet (prompt) is a
    distinct logical question (``question_id = set_<n>``). ``reference_answer``
    and ``question`` stay blank — the ASAP rubric/prompt text lives in a
    separate PDF not redistributed by the mirror.
    """
    cfg = cfg or load_data_config()
    ds = cfg.datasets["asap_sas"]
    asap_dir = cfg.paths.raw / ds.raw_subdir
    if not asap_dir.exists() or not ds.enabled:
        log.info("asap_sas not present or disabled — returning empty DataFrame.")
        return _coerce(pd.DataFrame())

    frames: list[pd.DataFrame] = []
    for fname, split_label in _ASAP_SPLIT_FILES.items():
        fpath = asap_dir / fname
        if not fpath.exists():
            continue
        raw = pd.read_csv(fpath, sep="\t")
        if raw.empty:
            continue
        sets = raw["EssaySet"].astype(str)
        frames.append(pd.DataFrame({
            "question_id": "set_" + sets,
            "question": "",
            "reference_answer": "",
            "student_answer": raw["EssayText"].astype(str),
            "score": pd.to_numeric(raw.get("Score1"), errors="coerce"),
            "label": "",
            "dataset": "asap_sas",
            "domain": sets.map(_ASAP_DOMAIN).fillna("science"),
            "split": split_label,
        }))
    if not frames:
        raise FileNotFoundError(f"ASAP-SAS: no split tsv under {asap_dir}.")
    out = pd.concat(frames, ignore_index=True)
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
    if cfg.datasets.get("powergrading") and cfg.datasets["powergrading"].enabled:
        out["powergrading"] = load_powergrading(cfg)
    if cfg.datasets.get("mindreading") and cfg.datasets["mindreading"].enabled:
        out["mindreading"] = load_mindreading(cfg)
    return out
