"""Phase 2C CLI — train & evaluate the late-fusion GBM head.

Runs the per-dataset evaluation protocol (official splits or k-fold, both over the
config seeds) and writes ``reports/phase2c/{results.json,results.csv,
feature_importance.json}`` plus summary figures.

Usage::

    python -m asag.models.train                 # all datasets with features.parquet
    python -m asag.models.train mohler asap_sas # only these
"""

from __future__ import annotations

import sys

from asag.models.evaluate import run_all

if __name__ == "__main__":
    run_all(only=sys.argv[1:] or None)
