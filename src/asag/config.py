"""Pydantic-backed configuration loader for `configs/data.yaml`.

Loading is centralized so paths and seeds stay consistent across modules and
runs. Call :func:`load_data_config` once at the start of any entrypoint.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Paths(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_root: Path
    raw: Path
    interim: Path
    processed: Path
    external: Path
    reports: Path
    figures: Path
    checksums: Path


class DatasetCfg(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool
    name: str
    raw_subdir: str
    license: str
    citation: str = ""


class EncoderViewCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    normalize_unicode: str
    collapse_whitespace: bool
    lowercase: bool
    remove_punctuation: bool
    remove_stopwords: bool


class NegationScopeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    window: int = Field(default=4, ge=1)
    marker: str = "prefix"  # prefix -> neg_word ; bracket reserved for future


class FeatureViewCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spacy_model: str
    lemmatize: bool
    lowercase: bool
    remove_punctuation: bool
    remove_stopwords: bool
    preserve_negators: list[str]
    negation_scope: NegationScopeCfg = Field(default_factory=NegationScopeCfg)


class PreprocessingCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    encoder_view: EncoderViewCfg
    feature_view: FeatureViewCfg


class SplitsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cv_k_folds: int = Field(ge=2)
    stratify_on: str


class ValidationCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    near_duplicate_jaccard_threshold: float = Field(ge=0.0, le=1.0)


# --- Phase 2B: feature engineering ---------------------------------------

class BranchFlagsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lexical: bool = True
    tfidf: bool = True
    negation: bool = True
    entities: bool = True
    semantic: bool = True
    rubric: bool = True


class TfidfCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ngram_min: int = Field(default=1, ge=1)
    ngram_max: int = Field(default=2, ge=1)
    min_df: int = Field(default=2, ge=1)


class RubricCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tau: float = Field(default=0.5, ge=0.0, le=1.0)


class NerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spacy_model: str = "en_core_web_sm"


class SemanticCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int = Field(default=64, ge=1)
    normalize: bool = True
    save_interaction_vector: bool = False
    use_cache: bool = True


class FeaturesCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    branches: BranchFlagsCfg = Field(default_factory=BranchFlagsCfg)
    tfidf: TfidfCfg = Field(default_factory=TfidfCfg)
    rubric: RubricCfg = Field(default_factory=RubricCfg)
    ner: NerCfg = Field(default_factory=NerCfg)
    semantic: SemanticCfg = Field(default_factory=SemanticCfg)


# --- Phase 2C: model selection & architecture ---------------------------

class LightGBMCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_estimators: int = Field(default=300, ge=1)
    learning_rate: float = Field(default=0.05, gt=0.0)
    num_leaves: int = Field(default=31, ge=2)
    min_child_samples: int = Field(default=20, ge=1)
    # Phase 2D regularization. Defaults (1.0 / 1.0 / 0 / 0) reproduce the Phase 2C
    # head exactly — the GBM stays deterministic, so re-running 2C is unchanged.
    # Optuna (Phase 2D) tunes subsample/colsample < 1, which makes the per-seed
    # std honestly non-zero (the bagging RNG depends on the seed).
    subsample: float = Field(default=1.0, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=1.0, gt=0.0, le=1.0)
    reg_alpha: float = Field(default=0.0, ge=0.0)
    reg_lambda: float = Field(default=0.0, ge=0.0)


class OrdinalCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # regression_threshold: train a regressor, then round-and-clip predictions
    # to the integer grade range observed in the training fold (QWK-friendly).
    strategy: str = "regression_threshold"


# --- Phase 2D: rigorous training (HPO + significance) -------------------

class HpoCfg(BaseModel):
    """Optuna hyperparameter search over the LightGBM head (Phase 2D).

    The objective uses only training-side data — the official ``dev`` split where
    one exists, otherwise an inner ``StratifiedKFold`` carved from the training
    rows. The held-out test splits are never touched during tuning.
    """
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    n_trials: int = Field(default=40, ge=1)
    inner_folds: int = Field(default=3, ge=2)
    timeout_s: int | None = None
    seed: int = 42


class SignificanceCfg(BaseModel):
    """Paired-bootstrap significance of the head vs the trivial baseline (Phase 2D)."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    n_boot: int = Field(default=10000, ge=100)
    ci: float = Field(default=0.95, gt=0.0, lt=1.0)
    seed: int = 42


class ModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    fusion_head: str = "lightgbm"
    seeds: list[int] = Field(default_factory=lambda: [42, 1, 2, 3, 4])
    lightgbm: LightGBMCfg = Field(default_factory=LightGBMCfg)
    ordinal: OrdinalCfg = Field(default_factory=OrdinalCfg)
    hpo: HpoCfg = Field(default_factory=HpoCfg)
    significance: SignificanceCfg = Field(default_factory=SignificanceCfg)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int
    paths: Paths
    datasets: dict[str, DatasetCfg]
    preprocessing: PreprocessingCfg
    splits: SplitsCfg
    validation: ValidationCfg
    features: FeaturesCfg = Field(default_factory=FeaturesCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)

    def project_root(self) -> Path:
        return getattr(self, "_root", Path.cwd().resolve())


def _find_project_root(start: Path | None = None) -> Path:
    """Locate the project root by walking upward looking for pyproject.toml.

    Search order:
      1. ``$ASAG_PROJECT_ROOT`` env var, if set and valid.
      2. Walk up from ``start`` (default: CWD), then from ``__file__``,
         until a directory containing ``pyproject.toml`` is found.
      3. Fall back to CWD as a last resort.
    """
    import os
    env = os.environ.get("ASAG_PROJECT_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "pyproject.toml").exists():
            return p

    for origin in [start or Path.cwd(), Path(__file__).resolve().parent]:
        cur = origin.resolve()
        for parent in [cur, *cur.parents]:
            if (parent / "pyproject.toml").exists() and (parent / "configs" / "data.yaml").exists():
                return parent

    return Path.cwd().resolve()


def _resolve_path(path_value: Any, root: Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (root / p)


@lru_cache(maxsize=4)
def load_data_config(cfg_path: Path | str | None = None) -> DataConfig:
    """Load and validate the data config. Cached by absolute path.

    If ``cfg_path`` is None we look for ``configs/data.yaml`` under the
    detected project root (see :func:`_find_project_root`).
    """
    if cfg_path:
        path = Path(cfg_path).expanduser().resolve()
        root = path.parent.parent if path.parent.name == "configs" else path.parent
    else:
        root = _find_project_root()
        path = root / "configs" / "data.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"configs/data.yaml not found at {path}. "
            "Set ASAG_PROJECT_ROOT or pass cfg_path explicitly."
        )

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    raw["paths"] = {k: _resolve_path(v, root) for k, v in raw["paths"].items()}
    cfg = DataConfig.model_validate(raw)
    # stash root for downstream consumers
    object.__setattr__(cfg, "_root", root)
    return cfg


def ensure_dirs(cfg: DataConfig) -> None:
    """Create all data/report directories if missing. Idempotent."""
    for p in (
        cfg.paths.raw,
        cfg.paths.interim,
        cfg.paths.processed,
        cfg.paths.external,
        cfg.paths.reports,
        cfg.paths.figures,
    ):
        p.mkdir(parents=True, exist_ok=True)
