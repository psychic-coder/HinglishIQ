"""Shared fixtures. No test touches the network or needs an API key."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backends import LLMRequest, LLMResponse  # noqa: E402
from src.cache import ResponseCache  # noqa: E402
from src.config import AppConfig, load_config  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.prompts import PromptSpec, load_prompt  # noqa: E402
from src.rate_limiter import RateLimiter  # noqa: E402

VALID_JSON = '{"intent": "refund_request", "entities": [{"type": "duration", "value": "3 days"}], "sentiment": "negative", "urgency": "high"}'


class FakeBackend:
    """Scripted backend: returns/raises the queued items in order."""

    def __init__(self, script: Sequence[str | BaseException]) -> None:
        self.script = list(script)
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return LLMResponse(text=item, input_tokens=100, output_tokens=20)


async def _no_sleep(_seconds: float) -> None:
    return None


def make_client(backend: FakeBackend, cache_dir: Path | None = None, max_retries: int = 3) -> LLMClient:
    """LLMClient wired for fast, deterministic tests."""
    return LLMClient(
        backend,
        model="test-model",
        temperature=0.0,
        max_output_tokens=256,
        rate_limiter=RateLimiter(0),
        max_concurrency=2,
        max_retries=max_retries,
        backoff_base_seconds=0.01,
        backoff_max_seconds=0.05,
        cache=ResponseCache(cache_dir) if cache_dir else None,
        sleep=_no_sleep,
    )


@pytest.fixture()
def prompt_v3() -> PromptSpec:
    return load_prompt(ROOT / "prompts" / "v3_schema_rules.yaml")


@pytest.fixture()
def tmp_config(tmp_path: Path) -> AppConfig:
    """Real config with every writable path redirected into ``tmp_path``."""
    raw = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    raw["paths"] = {
        "data_dir": str(tmp_path / "data"),
        "samples_file": str(ROOT / "data" / "samples.jsonl"),
        "prompts_dir": str(ROOT / "prompts"),
        "results_dir": str(tmp_path / "results"),
        "baseline_model": str(tmp_path / "models" / "baseline.joblib"),
    }
    raw["reliability"]["cache_dir"] = str(tmp_path / "cache")
    raw["mock"]["simulated_latency_ms"] = 0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_config(path)
