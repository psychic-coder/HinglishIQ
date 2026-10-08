"""Reliable structured-extraction client on top of an ``LLMBackend``.

Responsibilities: disk cache, client-side rate limiting, bounded concurrency,
exponential backoff with jitter on 429/5xx/RESOURCE_EXHAUSTED, Pydantic
validation with ONE repair retry, safe fallback, and per-call telemetry.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import ValidationError

from src.backends import LLMBackend, LLMRequest, LLMResponse, classify_error
from src.cache import ResponseCache, make_cache_key
from src.prompts import PromptSpec, render_system_instruction
from src.rate_limiter import LoopLocal, RateLimiter
from src.schemas import Extraction

logger = logging.getLogger(__name__)

Sleeper = Callable[[float], Awaitable[None]]

REPAIR_TEMPLATE = (
    "Your previous answer did not match the required JSON schema.\n"
    "Previous answer:\n{previous}\n\n"
    "Validation error:\n{error}\n\n"
    "Return ONLY a corrected JSON object for this customer message:\n{message}"
)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMCallError(RuntimeError):
    """Raised when a backend call fails permanently (after any retries)."""

    def __init__(self, message: str, retries: int, kind: str) -> None:
        super().__init__(message)
        self.retries = retries
        self.kind = kind  # "rate_limit" | "api"


@dataclass
class PredictionResult:
    """Outcome of one extraction call, with telemetry."""

    extraction: Extraction
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cache_hit: bool
    valid_json: bool
    retries: int
    repaired: bool = False
    error: str | None = None
    error_kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        data = asdict(self)
        data["extraction"] = self.extraction.model_dump()
        data["latency_ms"] = round(self.latency_ms, 2)
        return data


@dataclass
class ClientStats:
    """Cumulative counters for one client instance."""

    api_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retries: int = 0
    rate_limit_errors: int = 0
    repairs: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of requests served from cache."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0


def parse_extraction(text: str) -> tuple[Extraction | None, str | None]:
    """Validate raw model text against the schema; return (extraction, error)."""
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        return Extraction.model_validate_json(cleaned), None
    except ValidationError as exc:
        return None, str(exc)


class LLMClient:
    """Async-first extraction client with sync wrappers for CLI / Streamlit."""

    def __init__(
        self,
        backend: LLMBackend,
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
        rate_limiter: RateLimiter,
        max_concurrency: int,
        max_retries: int,
        backoff_base_seconds: float,
        backoff_max_seconds: float,
        cache: ResponseCache | None = None,
        thinking_budget: int | None = None,
        is_mock: bool = False,
        seed: int = 42,
        sleep: Sleeper = asyncio.sleep,
    ) -> None:
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.cache = cache
        self.is_mock = is_mock
        self.stats = ClientStats()
        self._rng = random.Random(seed)
        self._sleep = sleep
        self._semaphore: LoopLocal[asyncio.Semaphore] = LoopLocal(lambda: asyncio.Semaphore(max_concurrency))

    # ------------------------------------------------------------------ public
    def reset_stats(self) -> None:
        """Zero all counters (e.g. between evaluation methods)."""
        self.stats = ClientStats()

    async def aextract(self, text: str, prompt: PromptSpec) -> PredictionResult:
        """Extract structured fields from one (already preprocessed) message."""
        start = time.perf_counter()
        key = make_cache_key(self.model, self.temperature, prompt.content_hash, text)
        cached = self.cache.get(key) if self.cache else None
        if cached is not None:
            self.stats.cache_hits += 1
            return self._finish(
                PredictionResult(
                    extraction=Extraction.model_validate(cached["extraction"]),
                    latency_ms=0.0, input_tokens=0, output_tokens=0, cache_hit=True,
                    valid_json=bool(cached["valid_json"]), retries=0, repaired=bool(cached["repaired"]),
                ),
                start,
            )
        self.stats.cache_misses += 1
        request = LLMRequest(
            model=self.model,
            system_instruction=render_system_instruction(prompt),
            contents=text,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_budget=self.thinking_budget,
        )
        tokens = [0, 0]
        retries = 0
        try:
            response, retries = await self._call_with_retries(request)
            self._add_tokens(tokens, response)
            extraction, error = parse_extraction(response.text)
            repaired = False
            if extraction is None:
                repaired = True
                self.stats.repairs += 1
                logger.info("Invalid JSON from model, attempting one repair: %s", error)
                repair_request = LLMRequest(
                    **{**asdict(request), "contents": REPAIR_TEMPLATE.format(
                        previous=response.text, error=error, message=text)}
                )
                repair_response, extra = await self._call_with_retries(repair_request)
                retries += extra
                self._add_tokens(tokens, repair_response)
                extraction, error = parse_extraction(repair_response.text)
        except LLMCallError as exc:
            self.stats.failures += 1
            logger.warning("LLM call failed: %s", exc)
            return self._finish(
                PredictionResult(
                    extraction=Extraction.fallback(), latency_ms=0.0,
                    input_tokens=tokens[0], output_tokens=tokens[1], cache_hit=False,
                    valid_json=False, retries=exc.retries + retries, error=str(exc), error_kind=exc.kind,
                ),
                start,
            )
        valid = extraction is not None
        if not valid:
            logger.warning("Repair failed; falling back to intent 'other': %s", error)
        result = PredictionResult(
            extraction=extraction if extraction is not None else Extraction.fallback(),
            latency_ms=0.0, input_tokens=tokens[0], output_tokens=tokens[1], cache_hit=False,
            valid_json=valid, retries=retries, repaired=repaired,
        )
        if self.cache:
            self.cache.set(key, {
                "extraction": result.extraction.model_dump(),
                "valid_json": valid,
                "repaired": repaired,
                "model": self.model,
                "prompt_version": prompt.version,
            })
        return self._finish(result, start)

    async def aextract_many(self, texts: Sequence[str], prompt: PromptSpec) -> list[PredictionResult]:
        """Extract many messages concurrently (bounded by semaphore + rate limiter)."""
        return list(await asyncio.gather(*(self.aextract(t, prompt) for t in texts)))

    def extract(self, text: str, prompt: PromptSpec) -> PredictionResult:
        """Sync wrapper around :meth:`aextract` (must not be called inside a running loop)."""
        return asyncio.run(self.aextract(text, prompt))

    def extract_many(self, texts: Sequence[str], prompt: PromptSpec) -> list[PredictionResult]:
        """Sync wrapper around :meth:`aextract_many`."""
        return asyncio.run(self.aextract_many(texts, prompt))

    # ----------------------------------------------------------------- private
    def _finish(self, result: PredictionResult, start: float) -> PredictionResult:
        result.latency_ms = (time.perf_counter() - start) * 1000.0
        self.stats.latencies_ms.append(result.latency_ms)
        return result

    def _add_tokens(self, tokens: list[int], response: LLMResponse) -> None:
        tokens[0] += response.input_tokens
        tokens[1] += response.output_tokens
        self.stats.input_tokens += response.input_tokens
        self.stats.output_tokens += response.output_tokens

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with additive jitter, capped at backoff_max."""
        base = self.backoff_base_seconds * (2 ** attempt)
        jitter = self._rng.uniform(0, self.backoff_base_seconds)
        return min(self.backoff_max_seconds, base + jitter)

    async def _call_with_retries(self, request: LLMRequest) -> tuple[LLMResponse, int]:
        """Call the backend; retry transient failures. Returns (response, retries)."""
        attempt = 0
        while True:
            async with self._semaphore.get():
                await self.rate_limiter.acquire()
                self.stats.api_calls += 1
                try:
                    return await self.backend.generate(request), attempt
                except Exception as exc:  # noqa: BLE001 - classified below
                    retryable, rate_limited = classify_error(exc)
                    error_text = f"{type(exc).__name__}: {exc}"
                    if rate_limited:
                        self.stats.rate_limit_errors += 1
                    if not retryable or attempt >= self.max_retries:
                        kind = "rate_limit" if rate_limited else "api"
                        raise LLMCallError(error_text, retries=attempt, kind=kind) from exc
            delay = self._backoff_delay(attempt)
            attempt += 1
            self.stats.retries += 1
            logger.info("Transient API error (%s); retry %d in %.1fs", error_text, attempt, delay)
            await self._sleep(delay)
