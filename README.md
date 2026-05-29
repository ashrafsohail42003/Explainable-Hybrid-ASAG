# ASAG Research Project — Phase 1 (Data & Environment)

Research-grade **Automatic Short Answer Grading (ASAG)** system targeting publication (Q2/Q3).
Methodology: SBERT/DeBERTa semantic representations + interpretable linguistic features + rubric-aware concept coverage → **ordinal-regression** head. Cross-domain evaluation with telecommunications/networking (SAF) as the explainability case study.

**Scope (fixed):** short English answers (1–5 lines), content questions with reference answer + rubric, text-only (no OCR).

This branch covers **Phase 1 only**: reproducible environment, dataset acquisition (free sources), EDA, validation, and a two-view preprocessing pipeline. No modeling/training yet.

---

## Repository Layout

```
configs/                  # data.yaml: paths + seed + per-dataset flags
data/{raw,interim,processed,external}/   # gitignored; populated by `make download`
src/asag/
  config.py               # pydantic config loader
  data/
    download.py           # idempotent acquisition + sha256
    loaders.py            # unified-schema loaders
    preprocess.py         # two-views pipeline
    validate.py           # leakage/dup/missing/schema checks
    splits.py             # official splits + stratified k=5 scaffold
  utils/{seed.py, logging.py}
notebooks/01_eda.ipynb
reports/{phase1_report.md, DATASETS.md, figures/}
tests/test_loaders.py
```

## Reproducible Setup (from scratch)

### Requirements
- **Python 3.11** (installed automatically by `uv`)
- **uv** (https://docs.astral.sh/uv/ — install via `pipx install uv` or `winget install astral-sh.uv`)
- ~3 GB free disk for datasets + venv

### One-time setup

**With make (preferred):**

```bash
make setup
```

**Without make (Windows PowerShell equivalent):**

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$env:USERPROFILE\.cache\asag-venvs\asag-py311"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
uv python install 3.11
uv venv --python 3.11 $env:UV_PROJECT_ENVIRONMENT
uv pip install . --python "$env:UV_PROJECT_ENVIRONMENT\Scripts\python.exe"
& "$env:UV_PROJECT_ENVIRONMENT\Scripts\python.exe" -m spacy download en_core_web_sm
```

> **Why the venv lives outside the project on Windows**: this project's path
> includes Arabic characters. Python 3.11 reads venv `.pth` files via the
> system code page (cp1252) and fails on non-cp1252 bytes — the venv won't
> start. Keeping `.venv` at an ASCII path (under `~/.cache/asag-venvs/`)
> avoids the bug. `make setup` does this automatically.

### Dataset acquisition

```bash
make download
```

This downloads **SemEval-2013 Task 7**, **SAF Communication Networks English**, and **Mohler 2011** with `sha256` verification. Re-running is a no-op if checksums match.

**ASAP-SAS (optional, stretch goal)** — gated by Kaggle:

1. Create a Kaggle account at https://www.kaggle.com/
2. Accept competition rules at https://www.kaggle.com/competitions/asap-sas/rules
3. Place credentials at `~/.kaggle/kaggle.json` (chmod 600)
4. Flip `asap_sas.enabled: true` in `configs/data.yaml`
5. Re-run `make download`

### Validation, EDA, tests

```bash
make validate   # emits JSON reports under reports/
make eda        # executes notebooks/01_eda.ipynb
make test       # pytest smoke tests
```

## Reproducibility Notes

- Global seed `42` set in `src/asag/utils/seed.py` (covers `random`, `numpy`, `PYTHONHASHSEED`).
- All downloaded files have sha256 logged at `data/raw/CHECKSUMS.txt`.
- Dependencies pinned in `pyproject.toml`.
- See [reports/DATASETS.md](reports/DATASETS.md) for verified-link table, licenses, and citations.
- See [reports/phase1_report.md](reports/phase1_report.md) for decisions, EDA highlights, and risks.

## Datasets (Phase 1 stack)

| Dataset | Role | License |
|---|---|---|
| SemEval-2013 Task 7 (Beetle + SciEntsBank) | core + official UA/UQ/UD cross-domain splits | CC-BY-SA |
| SAF Communication Networks English | explainability case study (gold feedback) | CC-BY-4.0 |
| Mohler 2011 (CS data structures) | ordinal 0–5 grading; Kaggle mirror | research use |
| ASAP-SAS (Hewlett, 10 prompts) | rubric / QWK (optional, gated) | Kaggle competition terms |
| ASAG2024 (Meyerger) | cross-check only, not consumed | per source |

## License

This code is MIT licensed. **Each dataset retains its original license** — see [reports/DATASETS.md](reports/DATASETS.md).
