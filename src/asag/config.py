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
