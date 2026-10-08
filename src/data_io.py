"""JSONL reading/writing for labelled samples."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from src.schemas import LabeledSample


class DataError(RuntimeError):
    """Raised when a data file is missing or contains an invalid record."""


def load_samples(path: Path | str) -> list[LabeledSample]:
    """Load and validate every record of a JSONL file."""
    path = Path(path)
    if not path.is_file():
        raise DataError(f"Data file not found: {path}. Run `python main.py split` first.")
    samples: list[LabeledSample] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                samples.append(LabeledSample.model_validate_json(line))
            except ValidationError as exc:
                raise DataError(f"{path.name}:{line_no}: invalid sample: {exc}") from exc
    return samples


def write_samples(path: Path | str, samples: Iterable[LabeledSample]) -> int:
    """Write samples as JSONL; return the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")
            count += 1
    return count


def match_key(text: str) -> str:
    """Case/whitespace-insensitive key used for leakage checks."""
    return " ".join(text.lower().split())
