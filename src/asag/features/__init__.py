"""Phase 2B — feature engineering (late-fusion branches).

Each branch module exposes a pure function ``compute_*(df, cfg, ...) ->
pd.DataFrame`` returning row-aligned feature columns. ``build.py`` orchestrates
reads/writes; no branch performs I/O. See ``feature_dictionary.json`` (written
by the build) for the authoritative per-feature catalogue.
"""

from __future__ import annotations

FEATURES_SCHEMA_VERSION = "2b.1"

__all__ = ["FEATURES_SCHEMA_VERSION"]
