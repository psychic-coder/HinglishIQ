"""Retry/backoff, repair-retry, fallback and cache behaviour of LLMClient."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from conftest import VALID_JSON, FakeBackend, make_client

from src.backends import TransientAPIError, classify_error
from src.cache import make_cache_key


def test_retries_on_429_then_succeeds(prompt_v3):
    backend = FakeBackend([TransientAPIError(429, "RESOURCE_EXHAUSTED"), TransientAPIError(429), VALID_JSON])
    client = make_client(backend)
    result = client.extract("refund chahiye", prompt_v3)
    assert result.valid_json and result.error is None
    assert result.retries == 2
    assert client.stats.rate_limit_errors == 2
    assert client.stats.api_calls == 3
    assert result.extraction.intent == "refund_request"


def test_retries_on_5xx(prompt_v3):
    backend = FakeBackend([TransientAPIError(503, "unavailable"), VALID_JSON])
    result = make_client(backend).extract("x", prompt_v3)
    assert result.valid_json and result.retries == 1


def test_gives_up_after_max_retries(prompt_v3):
    backend = FakeBackend([TransientAPIError(429, "RESOURCE_EXHAUSTED")])
    client = make_client(backend, max_retries=2)
    result = client.extract("x", prompt_v3)
    assert result.error is not None and result.error_kind == "rate_limit"
    assert not result.valid_json
    assert result.extraction.intent == "other"
    assert client.stats.api_calls == 3  # 1 try + 2 retries


def test_non_retryable_error_fails_fast(prompt_v3):
    backend = FakeBackend([TransientAPIError(400, "API key not valid")])
    client = make_client(backend)
    result = client.extract("x", prompt_v3)
    assert result.error_kind == "api" and client.stats.api_calls == 1


def test_classify_error():
    assert classify_error(TransientAPIError(429)) == (True, True)
    assert classify_error(RuntimeError("RESOURCE_EXHAUSTED quota")) == (True, True)
    assert classify_error(TransientAPIError(500)) == (True, False)
    assert classify_error(TransientAPIError(400)) == (False, False)
    assert classify_error(TimeoutError()) == (True, False)


def test_backoff_is_exponential_and_capped():
    client = make_client(FakeBackend([VALID_JSON]))
    client.backoff_base_seconds, client.backoff_max_seconds = 1.0, 5.0
    delays = [client._backoff_delay(a) for a in range(5)]
    assert 1.0 <= delays[0] <= 2.0 and 2.0 <= delays[1] <= 3.0
    assert all(d <= 5.0 for d in delays)


def test_repair_retry_fixes_invalid_json(prompt_v3):
    backend = FakeBackend(['{"intent": "refund_request", "entities": [', VALID_JSON])
    client = make_client(backend)
    result = client.extract("refund chahiye", prompt_v3)
    assert result.valid_json and result.repaired
    assert client.stats.repairs == 1
    repair_contents = backend.requests[1].contents
    assert "did not match the required JSON schema" in repair_contents
    assert "refund chahiye" in repair_contents


def test_repair_failure_falls_back_and_flags_invalid(prompt_v3):
    bad_label = '{"intent": "angry_customer", "entities": [], "sentiment": "negative", "urgency": "high"}'
    backend = FakeBackend([bad_label, "not json at all"])
    result = make_client(backend).extract("x", prompt_v3)
    assert not result.valid_json and result.repaired
    assert result.extraction.intent == "other"
    assert len(backend.requests) == 2  # exactly ONE repair attempt


def test_code_fences_are_tolerated(prompt_v3):
    backend = FakeBackend([f"```json\n{VALID_JSON}\n```"])
    assert make_client(backend).extract("x", prompt_v3).valid_json


def test_cache_hit_and_miss(prompt_v3, tmp_path: Path):
    backend = FakeBackend([VALID_JSON])
    client = make_client(backend, cache_dir=tmp_path)
    first = client.extract("mera order kahan hai", prompt_v3)
    second = client.extract("mera order kahan hai", prompt_v3)
    assert not first.cache_hit and second.cache_hit
    assert len(backend.requests) == 1
    assert second.input_tokens == 0 and second.extraction == first.extraction
    assert client.stats.cache_hit_rate == 0.5


def test_cache_key_sensitive_to_prompt_content(prompt_v3, tmp_path: Path):
    backend = FakeBackend([VALID_JSON])
    client = make_client(backend, cache_dir=tmp_path)
    client.extract("same text", prompt_v3)
    edited = dataclasses.replace(prompt_v3, content_hash="different-file-content")  # same version label
    result = client.extract("same text", edited)
    assert not result.cache_hit and len(backend.requests) == 2


def test_cache_key_components():
    base = make_cache_key("m", 0.0, "h", "text")
    assert base == make_cache_key("m", 0.0, "h", "text")
    assert base != make_cache_key("m2", 0.0, "h", "text")
    assert base != make_cache_key("m", 0.5, "h", "text")
    assert base != make_cache_key("m", 0.0, "h2", "text")
    assert base != make_cache_key("m", 0.0, "h", "text2")


def test_failed_calls_are_not_cached(prompt_v3, tmp_path: Path):
    backend = FakeBackend([TransientAPIError(400, "bad"), VALID_JSON])
    client = make_client(backend, cache_dir=tmp_path)
    assert client.extract("x", prompt_v3).error is not None
    assert client.extract("x", prompt_v3).valid_json  # retried, not served from cache


def test_token_usage_recorded(prompt_v3):
    result = make_client(FakeBackend([VALID_JSON])).extract("x", prompt_v3)
    assert (result.input_tokens, result.output_tokens) == (100, 20)
    assert result.latency_ms >= 0


def test_extract_many_preserves_order(prompt_v3):
    client = make_client(FakeBackend([VALID_JSON]))
    results = client.extract_many(["a", "b", "c"], prompt_v3)
    assert len(results) == 3 and all(r.valid_json for r in results)
