"""End-to-end LLM stage: client construction, preprocessing and prediction."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

from dotenv import load_dotenv

from src.backends import GeminiBackend, LLMBackend
from src.cache import ResponseCache
from src.config import PROJECT_ROOT, AppConfig
from src.llm_client import LLMClient, PredictionResult
from src.mock_llm import MockBackend
from src.preprocess import PreprocessResult, preprocess
from src.prompts import PromptSpec
from src.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

API_KEY_ENV = "GEMINI_API_KEY"


class MissingAPIKeyError(RuntimeError):
    """Raised when real mode is requested but GEMINI_API_KEY is not set."""


@dataclass(frozen=True)
class PipelinePrediction:
    """A message, its preprocessing, and the LLM result."""

    original: str
    preprocessed: PreprocessResult
    result: PredictionResult


def get_api_key() -> str | None:
    """Return GEMINI_API_KEY from the environment / ``.env`` (or None)."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def build_llm_client(
    config: AppConfig,
    mock: bool | None = None,
    backend: LLMBackend | None = None,
) -> LLMClient:
    """Create an ``LLMClient`` from config.

    Args:
        config: validated application config.
        mock: True forces the mock backend, False forces Gemini, None means
            "real if a key exists, otherwise raise" (never silently fake).
        backend: optional explicit backend (tests); overrides ``mock``.

    Raises:
        MissingAPIKeyError: real mode requested without GEMINI_API_KEY.
    """
    use_mock = bool(mock)
    if backend is None:
        if use_mock:
            backend = MockBackend(config.mock.simulated_latency_ms)
        else:
            api_key = get_api_key()
            if api_key is None:
                raise MissingAPIKeyError(
                    f"{API_KEY_ENV} is not set. Copy .env.example to .env and add your free "
                    "Google AI Studio key, or run with --mock."
                )
            backend = GeminiBackend(api_key)
    reliability = config.reliability
    rpm = config.mock.rate_limit_rpm if use_mock else reliability.rate_limit_rpm
    return LLMClient(
        backend,
        model=config.mock.model_name if use_mock else config.llm.model,
        temperature=config.llm.temperature,
        max_output_tokens=config.llm.max_output_tokens,
        thinking_budget=config.llm.thinking_budget,
        rate_limiter=RateLimiter(rpm),
        max_concurrency=reliability.max_concurrency,
        max_retries=reliability.max_retries,
        backoff_base_seconds=reliability.backoff_base_seconds,
        backoff_max_seconds=reliability.backoff_max_seconds,
        cache=ResponseCache(reliability.cache_dir),
        is_mock=use_mock,
        seed=config.random_seed,
    )


async def apredict_many(client: LLMClient, prompt: PromptSpec, texts: Sequence[str]) -> list[PipelinePrediction]:
    """Preprocess and extract a batch of raw messages concurrently."""
    prepped = [preprocess(t) for t in texts]
    results = await client.aextract_many([p.text for p in prepped], prompt)
    return [PipelinePrediction(t, p, r) for t, p, r in zip(texts, prepped, results)]


def predict_many(client: LLMClient, prompt: PromptSpec, texts: Sequence[str]) -> list[PipelinePrediction]:
    """Sync wrapper around :func:`apredict_many`."""
    return asyncio.run(apredict_many(client, prompt, texts))


def predict(client: LLMClient, prompt: PromptSpec, text: str) -> PipelinePrediction:
    """Preprocess and extract a single raw message (sync)."""
    return predict_many(client, prompt, [text])[0]

