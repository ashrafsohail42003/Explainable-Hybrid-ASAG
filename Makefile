# ASAG Phase 1 — Makefile
# Works with GNU make. On Windows install via choco/scoop, or use the PowerShell equivalents documented in README.
#
# WHY UV_PROJECT_ENVIRONMENT points to %USERPROFILE%\.cache\asag-venvs:
# This project's path contains non-ASCII characters (Arabic). Python 3.x on
# Windows reads venv .pth files using the system code page (cp1252), which
# fails on non-cp1252 path bytes — Python crashes at venv startup. Keeping
# the venv at an ASCII path sidesteps the bug entirely.

PYTHON ?= python
UV ?= uv
export UV_PROJECT_ENVIRONMENT ?= $(USERPROFILE)/.cache/asag-venvs/asag-py311
export PYTHONUTF8 = 1
export PYTHONIOENCODING = utf-8

.PHONY: help setup download eda validate preprocess tokenstats features featuresvalidate train train2d xai test clean check

help:
	@echo "ASAG Phase 1 targets:"
	@echo "  make setup      - create venv (Python 3.11), install deps, download spaCy model"
	@echo "  make download   - run idempotent dataset downloads (SemEval + SAF + Mohler; ASAP-SAS optional)"
	@echo "  make validate   - run schema/leakage/duplicate validation, emit JSON reports"
	@echo "  make tokenstats - Phase 2A subword token-length study (justifies max_len)"
	@echo "  make features   - Phase 2B build feature matrices -> data/processed/<name>/features.parquet"
	@echo "  make featuresvalidate - Phase 2B feature-validation report + figures"
	@echo "  make train      - Phase 2C train+eval late-fusion GBM head -> reports/phase2c/"
	@echo "  make train2d    - Phase 2D Optuna HPO + paired-bootstrap + IAA ceiling -> reports/phase2d/"
	@echo "  make xai        - Phase 2F SHAP + concept coverage + SAF gold-feedback validation -> reports/phase2f/"
	@echo "  make eda        - execute notebooks/01_eda.ipynb and save figures"
	@echo "  make test       - run pytest smoke tests"
	@echo "  make check      - lint-light: import & schema sanity"
	@echo "  make clean      - remove caches (keeps data/raw)"

VENV_PY = $(UV_PROJECT_ENVIRONMENT)/Scripts/python.exe

setup:
	$(UV) python install 3.11
	$(UV) venv --python 3.11 "$(UV_PROJECT_ENVIRONMENT)"
	$(UV) pip install -e ".[dev]" --python "$(VENV_PY)"
	"$(VENV_PY)" -m spacy download en_core_web_sm

download:
	"$(VENV_PY)" -m asag.data.download

validate:
	"$(VENV_PY)" -m asag.data.validate

preprocess:
	"$(VENV_PY)" -m asag.data.preprocess

tokenstats:
	"$(VENV_PY)" -m asag.data.token_stats

features:
	"$(VENV_PY)" -m asag.features.build

featuresvalidate:
	"$(VENV_PY)" -m asag.features.validate_features

train:
	"$(VENV_PY)" -m asag.models.train

train2d:
	"$(VENV_PY)" -m asag.models.train2d

xai:
	"$(VENV_PY)" -m asag.xai.run

eda:
	"$(VENV_PY)" -m jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb

test:
	"$(VENV_PY)" -m pytest -q

check:
	"$(VENV_PY)" -c "import asag; from asag.config import load_data_config; load_data_config(); print('OK')"

clean:
	@echo "Removing caches (data/raw is preserved)..."
	"$(VENV_PY)" -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ipynb_checkpoints', 'src/asag/__pycache__'] ]; print('done')"
