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
    """Pull Mohler from Kaggle mirror + cross-check copy from ASAG2024."""
    ds = cfg.datasets["mohler"]
    if not ds.enabled:
        log.info("mohler disabled — skipping")
        return {"status": "skipped"}

    out_dir = cfg.paths.raw / ds.raw_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    kaggle_id = ds.model_extra.get("kaggle_dataset") if ds.model_extra else None
    assert kaggle_id, "configs/data.yaml: mohler kaggle_dataset missing"

    # primary: kaggle mirror
    marker = out_dir / ".kaggle_downloaded"
    if not marker.exists():
        log.info(f"mohler: downloading via kaggle CLI ({kaggle_id})")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", kaggle_id,
                 "-p", str(out_dir), "--unzip"],
                check=True, capture_output=True, text=True,
            )
            marker.write_text("ok")
        except FileNotFoundError:
            log.error(
                "Kaggle CLI not found. Install with `pip install kaggle`, "
                "then place ~/.kaggle/kaggle.json. Manual fallback: visit "
                f"https://www.kaggle.com/datasets/{kaggle_id} and unzip into {out_dir}."
            )
            return {"status": "manual_required", "instructions": kaggle_id}
        except subprocess.CalledProcessError as e:
            log.error(f"Kaggle CLI failed: {e.stderr}")
            log.error(
                "Ensure ~/.kaggle/kaggle.json exists with valid credentials, and that "
                f"the dataset {kaggle_id} is accessible to your account."
            )
            return {"status": "manual_required", "instructions": kaggle_id}
    else:
        log.info("mohler: kaggle download marker present — skipping")

    # log checksums for every file present
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != ".kaggle_downloaded":
            rel = str(p.relative_to(cfg.paths.raw)).replace("\\", "/")
            _write_checksum(cfg, rel, _sha256_file(p))

    # secondary: cross-check copy from ASAG2024
    crosscheck_hf = ds.model_extra.get("asag2024_crosscheck_hf_id") if ds.model_extra else None
    if crosscheck_hf:
        try:
            from datasets import load_dataset
            log.info(f"mohler: pulling cross-check subset from {crosscheck_hf}")
            cc = load_dataset(crosscheck_hf, split="train")
            cc_mohler = cc.filter(lambda r: (r.get("data_source") or "").lower() == "mohler")
            cc_path = out_dir / "asag2024_mohler_crosscheck.parquet"
            cc_mohler.to_parquet(str(cc_path))
            rel = str(cc_path.relative_to(cfg.paths.raw)).replace("\\", "/")
            _write_checksum(cfg, rel, _sha256_file(cc_path))
            log.info(f"mohler: cross-check wrote {cc_path} ({len(cc_mohler)} rows)")
        except Exception as e:
            log.warning(f"mohler cross-check failed (non-fatal): {e}")

    return {"status": "ok", "out_dir": str(out_dir)}


def download_asap_sas(cfg: DataConfig) -> dict:
    """Optional. Requires Kaggle credentials and accepted competition rules."""
    ds = cfg.datasets["asap_sas"]
    if not ds.enabled:
        log.info(
            "asap_sas disabled. To enable:\n"
            "  1) Create a Kaggle account.\n"
            "  2) Accept rules at https://www.kaggle.com/competitions/asap-sas/rules\n"
            "  3) Place credentials at ~/.kaggle/kaggle.json\n"
            "  4) Flip datasets.asap_sas.enabled to true in configs/data.yaml\n"
            "  5) Re-run `make download`"
        )
        return {"status": "manual_required"}

    comp = ds.model_extra.get("kaggle_competition") if ds.model_extra else None
    assert comp, "configs/data.yaml: asap_sas kaggle_competition missing"

    out_dir = cfg.paths.raw / ds.raw_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"asap_sas: downloading competition {comp} via kaggle CLI")
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", comp, "-p", str(out_dir)],
            check=True, capture_output=True, text=True,
        )
        # competition downloads ship as a single zip
        for z in out_dir.glob("*.zip"):
            _safe_extract_zip(z, out_dir / z.stem)
    except FileNotFoundError:
        log.error("Kaggle CLI not found — install `pip install kaggle`.")
        return {"status": "manual_required"}
    except subprocess.CalledProcessError as e:
        log.error(f"Kaggle CLI failed: {e.stderr}")
        log.error("If '403 Forbidden', accept the competition rules first.")
        return {"status": "manual_required"}

    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(cfg.paths.raw)).replace("\\", "/")
            _write_checksum(cfg, rel, _sha256_file(p))
    return {"status": "ok", "out_dir": str(out_dir)}


def download_asag2024(cfg: DataConfig) -> dict:
    """Pull the ASAG2024 unified benchmark for cross-checking only."""
    ds = cfg.datasets["asag2024"]
    if not ds.enabled:
        log.info("asag2024 disabled — skipping")
        return {"status": "skipped"}

    from datasets import load_dataset

    out_dir = cfg.paths.raw / ds.raw_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_id = ds.model_extra.get("hf_id") if ds.model_extra else None
    assert hf_id, "configs/data.yaml: asag2024 hf_id missing"

    log.info(f"asag2024: loading {hf_id} (cross-check only)")
    dsdict = load_dataset(hf_id)
    saved: dict[str, str] = {}
    for split_name, ds_split in dsdict.items():
        out_path = out_dir / f"{split_name}.parquet"
        ds_split.to_parquet(str(out_path))
        rel = str(out_path.relative_to(cfg.paths.raw)).replace("\\", "/")
        _write_checksum(cfg, rel, _sha256_file(out_path))
        saved[split_name] = str(out_path)
        log.info(f"asag2024: wrote {out_path} ({len(ds_split)} rows)")
    return {"status": "ok", "splits": saved}


# ---------- entrypoint ----------

def run_all(cfg: DataConfig) -> dict:
    ensure_dirs(cfg)
    set_global_seed(cfg.seed)
    results: dict[str, dict] = {}
    results["semeval"] = download_semeval(cfg)
    results["saf"] = download_saf(cfg)
    results["mohler"] = download_mohler(cfg)
    results["asap_sas"] = download_asap_sas(cfg)
    results["asag2024"] = download_asag2024(cfg)

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
