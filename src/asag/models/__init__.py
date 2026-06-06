"""Phase 2C — model selection & architecture.

The late-fusion head (``fusion.LgbmFusionHead``) is a NaN-native gradient-boosted
tree over the Phase 2B feature matrix. ``tasks`` declares the per-dataset task
type and evaluation protocol; ``evaluate`` runs the protocol over multiple seeds
and writes ``reports/phase2c/`` artifacts; ``train`` is the CLI entrypoint.
"""

from __future__ import annotations

MODELS_SCHEMA_VERSION = "2c.1"

__all__ = ["MODELS_SCHEMA_VERSION"]
