"""Phase 2F — Explainability (XAI) for the GBM late-fusion head.

Three deliverables, mirroring the consultation report's Phase 2F:

* ``shap_explain``        — SHAP over the fusion head (the *quantitative backbone*),
  computed with LightGBM's native TreeSHAP (no ``shap`` dependency).
* ``concept_attribution`` — per-rubric-concept coverage (the *pedagogical
  differentiator*): which reference concepts the student covered vs missed.
* ``saf_validation``      — the *novelty*: do the interpretable coverage signals
  align with SAF's human gold feedback (Correct / Partially correct / Incorrect)?

GBM-only slice — no torch. Outputs land in ``reports/phase2f/`` via
``python -m asag.xai.run``.
"""

from __future__ import annotations

XAI_SCHEMA_VERSION = "2f.1"

__all__ = ["XAI_SCHEMA_VERSION"]
