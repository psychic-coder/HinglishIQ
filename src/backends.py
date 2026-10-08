"""LLM transport backends: the real Gemini backend and the shared interface.

Concurrency design: orchestration (rate limiting, semaphore, retries) is
asyncio-based in ``llm_client``; the Gemini backend runs the SDK's *sync*
``client.models.generate_content`` in a worker thread via ``asyncio.to_thread``.
The sync HTTP client is not tied to any event loop, so the same backend can be
reused safely across repeated ``asyncio.run`` calls from the CLI and Streamlit.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from src.schemas import Extraction


def _gemini_response_schema() -> dict:
    """Return the extraction schema without JSON Schema keywords Gemini rejects."""
    def strip_unsupported(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: strip_unsupported(item)
                for key, item in value.items()
                if key != "additionalProperties"
            }
        if isinstance(value, list):
            return [strip_unsupported(item) for item in value]
        return value

    return strip_unsupported(Extraction.model_json_schema())  # type: ignore[return-value]


@dataclass(frozen=True)
class LLMRequest:
    """Everything a backend needs for one generation call."""

    model: str
    system_instruction: str
    contents: str
    temperature: float
    max_output_tokens: int
    thinking_budget: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Raw backend output plus token usage (0 when metadata is missing)."""

    text: str
    input_tokens: int
    output_tokens: int


class LLMBackend(Protocol):
    """Interface shared by the Gemini backend, the mock and test fakes."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Run one generation request."""
        ...


class TransientAPIError(RuntimeError):
    """A retryable API failure carrying an HTTP-like status code (used by fakes)."""

    def __init__(self, code: int, message: str = "") -> None:
        super().__init__(f"{code} {message}".strip())
        self.code = code


def classify_error(exc: BaseException) -> tuple[bool, bool]:
    """Return ``(retryable, rate_limited)`` for an exception from a backend.

    Retryable: HTTP 429, RESOURCE_EXHAUSTED, any 5xx, timeouts and connection
    errors. Everything else (bad request, invalid key, ...) fails fast.
    """
    code = getattr(exc, "code", None)
    if not isinstance(code, int):
        code = getattr(exc, "status_code", None)
    message = str(exc).upper()
    rate_limited = code == 429 or "RESOURCE_EXHAUSTED" in message
    server_error = isinstance(code, int) and 500 <= code < 600
    network = isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError))
    return (rate_limited or server_error or network, rate_limited)


def _usage_tokens(response: object) -> tuple[int, int]:
    """Read (input, output) token counts, guarding against missing metadata."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    prompt = getattr(usage, "prompt_token_count", None) or 0
    candidates = getattr(usage, "candidates_token_count", None) or 0
    thoughts = getattr(usage, "thoughts_token_count", None) or 0
    return int(prompt), int(candidates) + int(thoughts)


class GeminiBackend:
    """Google Gemini via the official ``google-genai`` SDK."""

    def __init__(self, api_key: str) -> None:
        from google import genai  # imported lazily so mock mode needs no SDK setup

        self._client = genai.Client(api_key=api_key)

    def _generate_sync(self, request: LLMRequest) -> LLMResponse:
        from google.genai import types

        config_kwargs: dict = {
            "system_instruction": request.system_instruction,
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
            "response_mime_type": "application/json",
            "response_schema": _gemini_response_schema(),
        }
        if request.thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=request.thinking_budget)
        response = self._client.models.generate_content(
            model=request.model,
            contents=request.contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        input_tokens, output_tokens = _usage_tokens(response)
        return LLMResponse(text=response.text or "", input_tokens=input_tokens, output_tokens=output_tokens)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Run the blocking SDK call in a worker thread."""
        return await asyncio.to_thread(self._generate_sync, request)
