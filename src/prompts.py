"""Versioned prompt files: loading, validation, discovery and rendering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.preprocess import preprocess
from src.schemas import Extraction

REQUIRED_FIELDS: tuple[str, ...] = ("version", "description", "changelog", "system_prompt", "few_shot")


class PromptError(ValueError):
    """Raised when a prompt file is missing, malformed or incomplete."""


@dataclass(frozen=True)
class FewShotExample:
    """One demonstration pair from a prompt file."""

    input: str
    output: Extraction


@dataclass(frozen=True)
class PromptSpec:
    """A parsed, validated prompt file.

    ``content_hash`` is the SHA-256 of the full file content; it is part of the
    LLM cache key so any edit to a prompt (even with the same version label)
    invalidates cached responses.
    """

    version: str
    description: str
    changelog: tuple[str, ...]
    system_prompt: str
    few_shot: tuple[FewShotExample, ...]
    content_hash: str
    path: Path


def _parse_changelog(value: object, path: Path) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return tuple(v.strip() for v in value)
    raise PromptError(f"{path.name}: 'changelog' must be a string or a list of strings")


def _parse_few_shot(value: object, path: Path) -> tuple[FewShotExample, ...]:
    if not isinstance(value, list):
        raise PromptError(f"{path.name}: 'few_shot' must be a list (use [] for zero-shot)")
    examples: list[FewShotExample] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or "input" not in item or "output" not in item:
            raise PromptError(f"{path.name}: few_shot[{index}] needs 'input' and 'output'")
        if not isinstance(item["input"], str) or not item["input"].strip():
            raise PromptError(f"{path.name}: few_shot[{index}].input must be a non-empty string")
        try:
            output = Extraction.model_validate(item["output"])
        except ValidationError as exc:
            raise PromptError(f"{path.name}: few_shot[{index}].output is invalid: {exc}") from exc
        examples.append(FewShotExample(input=item["input"], output=output))
    return tuple(examples)


def load_prompt(path: Path | str) -> PromptSpec:
    """Load and validate one prompt YAML file.

    Raises:
        PromptError: if the file is missing, not a mapping, lacks a required
            field, or contains an invalid few-shot example.
    """
    path = Path(path)
    if not path.is_file():
        raise PromptError(f"Prompt file not found: {path}")
    content = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise PromptError(f"{path.name}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise PromptError(f"{path.name}: top level must be a mapping")
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise PromptError(f"{path.name}: missing required field(s): {', '.join(missing)}")
    for field in ("version", "description", "system_prompt"):
        if not isinstance(data[field], str) or not data[field].strip():
            raise PromptError(f"{path.name}: '{field}' must be a non-empty string")
    return PromptSpec(
        version=data["version"].strip(),
        description=data["description"].strip(),
        changelog=_parse_changelog(data["changelog"], path),
        system_prompt=data["system_prompt"].strip(),
        few_shot=_parse_few_shot(data["few_shot"], path),
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        path=path,
    )


def discover_prompts(prompts_dir: Path | str) -> dict[str, Path]:
    """Map version label -> file path for every ``*.yaml`` in ``prompts_dir``.

    Raises:
        PromptError: if two files declare the same version.
    """
    found: dict[str, Path] = {}
    for path in sorted(Path(prompts_dir).glob("*.yaml")):
        spec = load_prompt(path)
        if spec.version in found:
            raise PromptError(f"Duplicate prompt version '{spec.version}' in {path.name}")
        found[spec.version] = path
    return found


def resolve_prompt(prompts_dir: Path | str, version: str) -> PromptSpec:
    """Load the prompt whose ``version`` field equals ``version``."""
    available = discover_prompts(prompts_dir)
    if version not in available:
        raise PromptError(
            f"Unknown prompt version '{version}'. Available: {', '.join(sorted(available)) or 'none'}"
        )
    return load_prompt(available[version])


def render_system_instruction(spec: PromptSpec) -> str:
    """Build the full system instruction: prompt text plus rendered examples.

    Few-shot inputs are passed through the same preprocessing as live inputs
    so demonstrations look exactly like what the model will receive.
    """
    if not spec.few_shot:
        return spec.system_prompt
    blocks = [spec.system_prompt, "", "EXAMPLES"]
    for number, example in enumerate(spec.few_shot, start=1):
        rendered = json.dumps(example.output.model_dump(), ensure_ascii=False)
        blocks.append(f"Example {number}\nMessage: {preprocess(example.input).text}\nJSON: {rendered}")
    return "\n\n".join(blocks)
