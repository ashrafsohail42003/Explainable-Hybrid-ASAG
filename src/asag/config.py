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


class FeatureViewCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spacy_model: str
    lemmatize: bool
    lowercase: bool
    remove_punctuation: bool
    remove_stopwords: bool
    preserve_negators: list[str]


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


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int
    paths: Paths
    datasets: dict[str, DatasetCfg]
    preprocessing: PreprocessingCfg
    splits: SplitsCfg
    validation: ValidationCfg

    def project_root(self) -> Path:
        return _PROJECT_ROOT


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CFG_PATH = _PROJECT_ROOT / "configs" / "data.yaml"


def _resolve(path_value: Any) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (_PROJECT_ROOT / p)


@lru_cache(maxsize=4)
def load_data_config(cfg_path: Path | str | None = None) -> DataConfig:
    """Load and validate the data config. Cached by absolute path."""
    path = Path(cfg_path).expanduser().resolve() if cfg_path else _DEFAULT_CFG_PATH
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    raw["paths"] = {k: _resolve(v) for k, v in raw["paths"].items()}
    return DataConfig.model_validate(raw)


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
