"""SBERT bi-encoder semantic features (``sem_*``).

A single ``SbertEncoder`` (all-MiniLM-L6-v2 by default) is shared with the
rubric branch so the model loads once. It dedups + caches embeddings in memory
(and optionally on disk) so each unique text is encoded once — the key to CPU
feasibility over ~44k rows.

Scalar features go into ``features.parquet``:
  * ``sem_cosine``         — cosine(student, reference)
  * ``sem_abs_diff_mean``  — mean(|u−v|)
  * ``sem_hadamard_mean``  — mean(u⊙v)
We deliberately omit ``sem_dot`` / ``sem_euclidean``: with normalized
embeddings they are algebraic restatements of cosine and add no signal.

The full 768-d interaction block (|u−v| ⊕ u⊙v) is returned separately and only
persisted by the build when ``features.semantic.save_interaction_vector`` is set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from asag.features.text_utils import NAN

SCALAR_COLUMNS = ["sem_cosine", "sem_abs_diff_mean", "sem_hadamard_mean"]


class SbertEncoder:
    """Lazy, deduped, cached SentenceTransformer wrapper."""

    def __init__(self, model_name: str, batch_size: int = 64, normalize: bool = True, log=None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.batch_size = batch_size
        self.normalize = normalize
        self.log = log
        self._cache: dict[str, int] = {}
        self._mat: np.ndarray | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        texts = [t if isinstance(t, str) else "" for t in texts]
        unseen = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if unseen:
            emb = self.model.encode(
                unseen, batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True, show_progress_bar=False,
            ).astype(np.float32)
            base = 0 if self._mat is None else self._mat.shape[0]
            self._mat = emb if self._mat is None else np.vstack([self._mat, emb])
            for i, t in enumerate(unseen):
                self._cache[t] = base + i
        return self._mat[[self._cache[t] for t in texts]]

    # --- optional cross-run disk cache (single file, model-name guarded) ---
    def load_cache(self, path: Path) -> None:
        if not path.exists():
            return
        data = np.load(path, allow_pickle=True)
        if str(data.get("model")) != self.model_name:
            return  # different model -> incompatible embeddings; ignore
        texts, mat = data["texts"].tolist(), data["emb"]
        rows = [mat[i] for i, t in enumerate(texts) if t not in self._cache]
        names = [t for t in texts if t not in self._cache]
        if rows:
            base = 0 if self._mat is None else self._mat.shape[0]
            add = np.vstack(rows).astype(np.float32)
            self._mat = add if self._mat is None else np.vstack([self._mat, add])
            for j, t in enumerate(names):
                self._cache[t] = base + j

    def save_cache(self, path: Path) -> None:
        if self._mat is None:
            return
        texts: list[str] = [""] * self._mat.shape[0]
        for t, i in self._cache.items():
            texts[i] = t
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, model=self.model_name, texts=np.array(texts, dtype=object), emb=self._mat)


def compute_semantic(df: pd.DataFrame, cfg, encoder: SbertEncoder):
    """Return (scalar_df, dense_block | None). dense_block is (n, 2*dim)."""
    students = df["student_answer_enc"].fillna("").astype(str).tolist()
    refs = df["reference_answer_enc"].fillna("").astype(str).tolist()
    has_ref = np.array([bool(r.strip()) for r in refs])

    u = encoder.embed(students)
    v = encoder.embed(refs)

    cosine = np.einsum("ij,ij->i", u, v).astype(np.float64)  # normalized -> dot == cos
    abs_diff = np.abs(u - v)
    hadamard = u * v
    scalars = pd.DataFrame({
        "sem_cosine": cosine,
        "sem_abs_diff_mean": abs_diff.mean(axis=1).astype(np.float64),
        "sem_hadamard_mean": hadamard.mean(axis=1).astype(np.float64),
    }, index=df.index)
    scalars.loc[~has_ref, SCALAR_COLUMNS] = NAN

    dense = None
    if cfg.features.semantic.save_interaction_vector:
        dense = np.concatenate([abs_diff, hadamard], axis=1).astype(np.float32)
        dense[~has_ref] = np.nan

    return scalars, dense
