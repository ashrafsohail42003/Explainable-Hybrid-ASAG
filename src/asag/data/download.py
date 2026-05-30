"""Idempotent dataset downloaders with sha256 verification.

Usage::

    python -m asag.data.download

Behavior:
  - SemEval-2013 Task 7: downloads two zips from the myrosia/semeval-2013-task7
    GitHub repo and extracts them under data/raw/semeval-2013-task7/.
  - SAF Communication Networks English: pulled via HuggingFace ``datasets``;
    cached parquet copy saved under data/raw/saf-comm-nets-en/.
  - Mohler 2011: pulled via the Kaggle CLI (mubeenfurqanahmed mirror); a
    cross-check copy is also pulled from the Meyerger/ASAG2024 HF dataset
    (filtered to the Mohler subset).
  - ASAP-SAS: only attempted if ``datasets.asap_sas.enabled`` is true in
    configs/data.yaml. Otherwise the script prints clear manual-step
    instructions and exits 0 for that dataset.
  - ASAG2024: HF download of the unified benchmark used for cross-checking.

All downloads are idempotent: a sha256 manifest is maintained at
data/raw/CHECKSUMS.txt and re-runs skip files whose hash already matches.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

from asag.config import DataConfig, ensure_dirs, load_data_config
from asag.utils.logging import get_logger
from asag.utils.seed import set_global_seed

log = get_logger()


# ---------- utilities ----------

def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_checksums(cfg: DataConfig) -> dict[str, str]:
    p = cfg.paths.checksums
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sha, _, rel = line.partition("  ")
        if sha and rel:
            out[rel] = sha
    return out


def _write_checksum(cfg: DataConfig, rel: str, sha: str) -> None:
    p = cfg.paths.checksums
    existing = _load_checksums(cfg)
    existing[rel] = sha
    lines = ["# sha256  relative-path-from-data/raw"]
    for k in sorted(existing):
        lines.append(f"{existing[k]}  {k}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _http_get(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> None:
    """Stream a URL to disk with retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            log.info(f"GET {url} (attempt {attempt}) -> {dest}")
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
            return
        except Exception as e:  # network: retry
            last_err = e
            log.warning(f"download failed: {e}; retrying ...")
    assert last_err is not None
    raise last_err


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            member_path = (dest_dir / member).resolve()
            if not str(member_path).startswith(str(dest_dir.resolve())):
                raise RuntimeError(f"unsafe zip member: {member}")
        z.extractall(dest_dir)


# ---------- per-dataset downloaders ----------

def download_semeval(cfg: DataConfig) -> dict:
    ds = cfg.datasets["semeval"]
    if not ds.enabled:
        log.info("semeval disabled — skipping")
        return {"status": "skipped"}

    out_dir = cfg.paths.raw / ds.raw_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    url_3way = ds.model_extra.get("url_3way") if ds.model_extra else None
    url_5way = ds.model_extra.get("url_5way") if ds.model_extra else None
    assert url_3way and url_5way, "configs/data.yaml: semeval url_3way/url_5way missing"

    targets = [(url_3way, out_dir / "semeval-3way.zip"),
               (url_5way, out_dir / "semeval-5way.zip")]

    checksums = _load_checksums(cfg)
    for url, dest in targets:
        rel = str(dest.relative_to(cfg.paths.raw)).replace("\\", "/")
        if dest.exists() and checksums.get(rel) == _sha256_file(dest):
            log.info(f"semeval: {rel} sha256 matches — skipping download")
            continue
        _http_get(url, dest)
        sha = _sha256_file(dest)
        _write_checksum(cfg, rel, sha)

        # extract into per-zip subdir
        extract_dir = out_dir / dest.stem
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        _safe_extract_zip(dest, extract_dir)
        log.info(f"semeval: extracted {dest.name} -> {extract_dir}")

    return {"status": "ok", "out_dir": str(out_dir)}


def download_saf(cfg: DataConfig) -> dict:
    """Pull SAF from HuggingFace and cache one parquet per split under data/raw."""
    ds = cfg.datasets["saf"]
    if not ds.enabled:
        log.info("saf disabled — skipping")
        return {"status": "skipped"}

    from datasets import load_dataset  # local import; heavy dep

    out_dir = cfg.paths.raw / ds.raw_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_id = ds.model_extra.get("hf_id") if ds.model_extra else None
    assert hf_id, "configs/data.yaml: saf hf_id missing"

    log.info(f"saf: loading {hf_id} via HuggingFace datasets")
    dsdict = load_dataset(hf_id)

    saved: dict[str, str] = {}
    for split_name, ds_split in dsdict.items():
        out_path = out_dir / f"{split_name}.parquet"
        ds_split.to_parquet(str(out_path))
        rel = str(out_path.relative_to(cfg.paths.raw)).replace("\\", "/")
        _write_checksum(cfg, rel, _sha256_file(out_path))
        saved[split_name] = str(out_path)
        log.info(f"saf: wrote {out_path} ({len(ds_split)} rows)")

    return {"status": "ok", "splits": saved}


def download_mohler(cfg: DataConfig) -> dict:
    """Pull canonical Mohler 2011 from the ASAG2024 unified benchmark.

    We learned during Phase 1 that the Kaggle mirror suggested in the plan
    (``mubeenfurqanahmed/automatic-short-answer-grading-dataset``) is NOT
    actual Mohler 2011 — it contains generic science short-answer items,
    not CS data-structures. The canonical source is ASAG2024's ``mohler``
    subset, which we extract here.
    """
    ds = cfg.datasets["mohler"]
    if not ds.enabled:
        log.info("mohler disabled — skipping")
        return {"status": "skipped"}

    out_dir = cfg.paths.raw / ds.raw_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    crosscheck_hf = ds.model_extra.get("asag2024_crosscheck_hf_id") if ds.model_extra else None
    if not crosscheck_hf:
        log.error("mohler: asag2024_crosscheck_hf_id missing in config")
        return {"status": "error", "reason": "missing config"}

    canonical_path = out_dir / "mohler_canonical_from_asag2024.parquet"
    if canonical_path.exists():
        log.info(f"mohler: canonical parquet present — skipping ({canonical_path})")
        return {"status": "ok", "out_dir": str(out_dir), "rows": "cached"}

    try:
        from huggingface_hub import HfFileSystem, hf_hub_download
        import pandas as pd
        log.info(f"mohler: pulling canonical subset from {crosscheck_hf}")
        fs = HfFileSystem()
        parquets = [Path(p).name for p in fs.glob(f"datasets/{crosscheck_hf}/**/*.parquet")]
        frames = []
        for fname in parquets or ["train.parquet", "validation.parquet", "test.parquet"]:
            try:
                local = hf_hub_download(repo_id=crosscheck_hf, filename=fname,
                                        repo_type="dataset")
            except Exception:
                continue
            df = pd.read_parquet(local)
            if "__index_level_0__" in df.columns:
                df = df.drop(columns=["__index_level_0__"])
            if "data_source" in df.columns:
                df = df[df["data_source"].str.lower() == "mohler"]
            if len(df):
                frames.append(df)
        if not frames:
            log.error("mohler: no rows recovered from ASAG2024")
            return {"status": "error", "reason": "no rows"}
        mohler_all = pd.concat(frames, ignore_index=True)
        mohler_all.to_parquet(canonical_path, index=False)
        rel = str(canonical_path.relative_to(cfg.paths.raw)).replace("\\", "/")
        _write_checksum(cfg, rel, _sha256_file(canonical_path))
        log.info(f"mohler: canonical wrote {canonical_path} ({len(mohler_all)} rows)")
        return {"status": "ok", "out_dir": str(out_dir), "rows": int(len(mohler_all))}
    except Exception as e:
        log.error(f"mohler canonical extraction failed: {e}")
        return {"status": "error", "reason": str(e)}


def download_asap_sas(cfg: DataConfig) -> dict:
    """Acquire ASAP-SAS via the freely redistributable AERA mirror.

    The official Kaggle competition data is gated behind manual rule
    acceptance, so we pull the AERA dataset (Li et al., Findings of EMNLP
    2023, CC-BY-NC-4.0) instead. It republishes the science/biology prompts
    (EssaySets 1, 2, 5, 6) with both human-rater scores and — unlike the
    original Kaggle release — gold scores on the test split too. We keep the
    canonical ASAP columns (plus ``llm_rationale`` for the Phase 2
    explainability study) and write one TSV per split under
    ``data/raw/asap-sas/`` so the loader stays a simple tabular reader.
    """
    ds = cfg.datasets["asap_sas"]
    if not ds.enabled:
        log.info("asap_sas disabled — skipping")
        return {"status": "skipped"}

    extra = ds.model_extra or {}
    hf_id = extra.get("mirror_hf_id")
    config = extra.get("mirror_config", "example")
    if not hf_id:
        log.error("asap_sas: mirror_hf_id missing in config — cannot download.")
        return {"status": "manual_required"}

    from huggingface_hub import hf_hub_download
    import pandas as pd

    out_dir = cfg.paths.raw / ds.raw_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_cols = ["Id", "EssaySet", "Score1", "Score2", "EssayText", "llm_rationale"]
    # AERA source split filename -> our raw split file (loader maps val->dev, test->test_ua)
    split_files = {"train": "train", "val": "dev", "test": "test"}
    saved: dict[str, str] = {}
    for src_split, out_name in split_files.items():
        try:
            local = hf_hub_download(
                repo_id=hf_id, filename=f"{config}/{src_split}.json",
                repo_type="dataset",
            )
        except Exception as e:
            log.warning(f"asap_sas: could not fetch {config}/{src_split}.json: {e}")
            continue
        with open(local, encoding="utf-8") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        for c in keep_cols:
            if c not in df.columns:
                df[c] = ""
        out_path = out_dir / f"{out_name}.tsv"
        # pandas quotes any field containing tab/newline, so the TSV round-trips safely.
        df[keep_cols].to_csv(out_path, sep="\t", index=False)
        rel = str(out_path.relative_to(cfg.paths.raw)).replace("\\", "/")
        _write_checksum(cfg, rel, _sha256_file(out_path))
        saved[out_name] = str(out_path)
        log.info(f"asap_sas: wrote {out_path} ({len(df)} rows)")

    if not saved:
        log.error("asap_sas: no split files retrieved from mirror.")
        return {"status": "manual_required"}
    return {"status": "ok", "out_dir": str(out_dir), "splits": saved}


def download_asag2024(cfg: DataConfig) -> dict:
    """Pull the ASAG2024 unified benchmark for cross-checking only.

    The HuggingFace dataset card declares a schema that doesn't quite
    match the on-disk parquet files (`__index_level_0__` ghost column),
    so ``load_dataset`` raises a CastError. We side-step that by pulling
    the parquet files directly via ``huggingface_hub.hf_hub_download``
    and reading them with pandas — the schema mismatch is irrelevant.
    """
    ds = cfg.datasets["asag2024"]
    if not ds.enabled:
        log.info("asag2024 disabled — skipping")
        return {"status": "skipped"}

    from huggingface_hub import HfFileSystem, hf_hub_download
    import pandas as pd

    out_dir = cfg.paths.raw / ds.raw_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_id = ds.model_extra.get("hf_id") if ds.model_extra else None
    assert hf_id, "configs/data.yaml: asag2024 hf_id missing"

    log.info(f"asag2024: discovering parquet files in {hf_id}")
    fs = HfFileSystem()
    parquet_files = [Path(p).name for p in fs.glob(f"datasets/{hf_id}/**/*.parquet")]
    if not parquet_files:
        log.warning(f"asag2024: no parquet files found via fs.glob; trying default splits")
        parquet_files = ["train.parquet", "validation.parquet", "test.parquet"]

    saved: dict[str, str] = {}
    for fname in parquet_files:
        try:
            local = hf_hub_download(repo_id=hf_id, filename=fname, repo_type="dataset")
        except Exception as e:
            # try data/<fname> subpath
            try:
                local = hf_hub_download(repo_id=hf_id, filename=f"data/{fname}",
                                        repo_type="dataset")
            except Exception:
                log.warning(f"asag2024: could not fetch {fname}: {e}")
                continue
        local = Path(local)
        df = pd.read_parquet(local)
        # drop ghost index column if present
        for ghost in ("__index_level_0__",):
            if ghost in df.columns:
                df = df.drop(columns=[ghost])
        out_path = out_dir / Path(fname).name
        df.to_parquet(out_path, index=False)
        rel = str(out_path.relative_to(cfg.paths.raw)).replace("\\", "/")
        _write_checksum(cfg, rel, _sha256_file(out_path))
        saved[Path(fname).stem] = str(out_path)
        log.info(f"asag2024: wrote {out_path} ({len(df)} rows)")
    return {"status": "ok", "splits": saved}


# ---------- entrypoint ----------

def download_powergrading(cfg: DataConfig) -> dict:
    """Pull the Powergrading 1.0 corpus from Microsoft Research download center."""
    ds = cfg.datasets.get("powergrading")
    if ds is None or not ds.enabled:
        log.info("powergrading disabled — skipping")
        return {"status": "skipped"}

    out_dir = cfg.paths.raw / ds.raw_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    url = ds.model_extra.get("url") if ds.model_extra else None
    assert url, "configs/data.yaml: powergrading url missing"

    zip_path = out_dir / "Powergrading-1.0-Corpus.zip"
    marker = out_dir / "README.txt"
    if marker.exists():
        log.info("powergrading: already extracted — skipping")
        return {"status": "ok", "out_dir": str(out_dir), "cached": True}

    _http_get(url, zip_path)
    rel = str(zip_path.relative_to(cfg.paths.raw)).replace("\\", "/")
    _write_checksum(cfg, rel, _sha256_file(zip_path))
    _safe_extract_zip(zip_path, out_dir)
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.suffix != ".zip":
            r = str(p.relative_to(cfg.paths.raw)).replace("\\", "/")
            _write_checksum(cfg, r, _sha256_file(p))
    log.info(f"powergrading: extracted to {out_dir}")
    return {"status": "ok", "out_dir": str(out_dir)}


def download_mindreading(cfg: DataConfig) -> dict:
    """Pull MIND-CA (Kovatchev 2020) xlsx files from the COLING 2020 GitHub release."""
    ds = cfg.datasets.get("mindreading")
    if ds is None or not ds.enabled:
        log.info("mindreading disabled — skipping")
        return {"status": "skipped"}

    out_dir = cfg.paths.raw / ds.raw_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    base = ds.model_extra.get("github_raw_base") if ds.model_extra else None
    files = ds.model_extra.get("files") if ds.model_extra else None
    assert base and files, "configs/data.yaml: mindreading github_raw_base + files missing"

    checksums = _load_checksums(cfg)
    n_downloaded = 0
    for fname in files:
        dest = out_dir / fname
        rel = str(dest.relative_to(cfg.paths.raw)).replace("\\", "/")
        if dest.exists() and checksums.get(rel) == _sha256_file(dest):
            log.info(f"mindreading: {fname} sha256 matches — skipping")
            continue
        _http_get(f"{base}/{fname}", dest)
        _write_checksum(cfg, rel, _sha256_file(dest))
        n_downloaded += 1
    log.info(f"mindreading: {n_downloaded} new files; {len(files)} total in {out_dir}")
    return {"status": "ok", "out_dir": str(out_dir), "files": len(files)}


def run_all(cfg: DataConfig) -> dict:
    ensure_dirs(cfg)
    set_global_seed(cfg.seed)
    results: dict[str, dict] = {}
    results["semeval"] = download_semeval(cfg)
    results["saf"] = download_saf(cfg)
    results["mohler"] = download_mohler(cfg)
    results["asap_sas"] = download_asap_sas(cfg)
    results["asag2024"] = download_asag2024(cfg)
    results["powergrading"] = download_powergrading(cfg)
    results["mindreading"] = download_mindreading(cfg)

    summary_path = cfg.paths.raw / "_download_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info(f"summary: {summary_path}")
    return results


if __name__ == "__main__":
    cfg = load_data_config()
    res = run_all(cfg)
    print(json.dumps(res, indent=2))
    failed = [k for k, v in res.items() if v.get("status") not in {"ok", "skipped", "manual_required"}]
    sys.exit(1 if failed else 0)
