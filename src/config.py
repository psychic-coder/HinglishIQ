"""Typed, validated configuration loader."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMSettings(_Strict):
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    max_output_tokens: int = Field(gt=0)
    thinking_budget: int | None = None

    @model_validator(mode="after")
    def _no_pro_models(self) -> "LLMSettings":
        if re.search(r"(^|[-_/])pro($|[-_])", self.model.lower()):
            raise ValueError("Pro models are not free-tier; use a Flash / Flash-Lite model.")
        return self


class ReliabilitySettings(_Strict):
    rate_limit_rpm: int = Field(ge=0)
    max_concurrency: int = Field(ge=1)
    max_retries: int = Field(ge=0)
    backoff_base_seconds: float = Field(ge=0.0)
    backoff_max_seconds: float = Field(ge=0.0)
    cache_dir: Path


class PathSettings(_Strict):
    data_dir: Path
    samples_file: Path
    prompts_dir: Path
    results_dir: Path
    baseline_model: Path


class SplitSettings(_Strict):
    train: float = Field(gt=0, lt=1)
    dev: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _sums_to_one(self) -> "SplitSettings":
        if abs(self.train + self.dev + self.test - 1.0) > 1e-6:
            raise ValueError("split ratios must sum to 1.0")
        return self


class EvaluationSettings(_Strict):
    codemix_low_max: float = Field(ge=0, le=1)
    codemix_med_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> "EvaluationSettings":
        if self.codemix_low_max >= self.codemix_med_max:
            raise ValueError("codemix_low_max must be < codemix_med_max")
        return self


class BaselineSettings(_Strict):
    ngram_min: int = Field(ge=1)
    ngram_max: int = Field(ge=1)
    C: float = Field(gt=0)
    max_iter: int = Field(gt=0)


class MockSettings(_Strict):
    model_name: str
    rate_limit_rpm: int = Field(ge=0)
    simulated_latency_ms: int = Field(ge=0)


class AppConfig(_Strict):
    """Root configuration object. All paths are absolute after loading."""

    llm: LLMSettings
    reliability: ReliabilitySettings
    paths: PathSettings
    split: SplitSettings
    random_seed: int
    default_prompt_version: str
    evaluation: EvaluationSettings
    baseline: BaselineSettings
    mock: MockSettings

    def split_file(self, split: str) -> Path:
        """Return the JSONL path of a data split (train/dev/test)."""
        return self.paths.data_dir / f"{split}.jsonl"


def _absolutize(raw: dict, root: Path) -> dict:
    paths = dict(raw.get("paths", {}))
    for key, value in paths.items():
        paths[key] = str((root / value).resolve()) if not Path(value).is_absolute() else value
    raw["paths"] = paths
    reliability = dict(raw.get("reliability", {}))
    if "cache_dir" in reliability and not Path(reliability["cache_dir"]).is_absolute():
        reliability["cache_dir"] = str((root / reliability["cache_dir"]).resolve())
    raw["reliability"] = reliability
    return raw


def load_config(path: Path | str = DEFAULT_CONFIG_PATH, root: Path | None = None) -> AppConfig:
    """Load and validate the YAML config; relative paths resolve against ``root``.

    ``root`` defaults to the project root (parent of the ``config/`` directory).
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    root = root or path.resolve().parent.parent
    try:
        return AppConfig.model_validate(_absolutize(raw, root))
    except ValidationError as exc:
        raise ConfigError(f"Invalid config {path}:\n{exc}") from exc
