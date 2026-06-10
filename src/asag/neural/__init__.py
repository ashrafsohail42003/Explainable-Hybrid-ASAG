"""Phase 2G — the neural slice (DeBERTa cross-encoder + CORAL/CORN).

End-to-end fine-tuned transformer grading, the credibility baseline the GBM-only
slices deferred. Torch is imported lazily everywhere so the rest of the package
(and the test suite) keeps working when torch is absent.
"""

from __future__ import annotations

import importlib.util

NEURAL_SCHEMA_VERSION = "2g.1"

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
