# ASAG Phase 1 — Makefile
# Works with GNU make. On Windows install via choco/scoop, or use the PowerShell equivalents documented in README.

PYTHON ?= python
UV ?= uv

.PHONY: help setup download eda validate test clean check

help:
	@echo "ASAG Phase 1 targets:"
	@echo "  make setup      - create venv (Python 3.11), install deps, download spaCy model"
	@echo "  make download   - run idempotent dataset downloads (SemEval + SAF + Mohler; ASAP-SAS optional)"
	@echo "  make validate   - run schema/leakage/duplicate validation, emit JSON reports"
	@echo "  make eda        - execute notebooks/01_eda.ipynb and save figures"
	@echo "  make test       - run pytest smoke tests"
	@echo "  make check      - lint-light: import & schema sanity"
	@echo "  make clean      - remove caches (keeps data/raw)"

setup:
	$(UV) python install 3.11
	$(UV) venv --python 3.11
	$(UV) pip install -e ".[dev]"
	$(UV) run python -m spacy download en_core_web_sm

download:
	$(UV) run python -m asag.data.download

validate:
	$(UV) run python -m asag.data.validate

eda:
	$(UV) run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb

test:
	$(UV) run pytest -q

check:
	$(UV) run python -c "import asag; from asag.config import load_data_config; load_data_config(); print('OK')"

clean:
	@echo "Removing caches (data/raw is preserved)..."
	$(UV) run python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ipynb_checkpoints', 'src/asag/__pycache__'] ]; print('done')"
