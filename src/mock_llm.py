"""Deterministic fake LLM backend for ``--mock`` mode, tests and the demo UI.

It implements the same ``LLMBackend`` interface as the Gemini backend and
returns schema-valid JSON produced by transparent keyword rules plus the regex
entity extractor. Results are FAKE: they exist so the whole pipeline (cache,
rate limiter, evaluation, UI) runs with no API key and no network.
"""

from __future__ import annotations

import asyncio
import json
import re

from src.backends import LLMRequest, LLMResponse
from src.entities import extract_entities
from src.schemas import Extraction

_INTENT_RULES: tuple[tuple[str, str], ...] = (
    ("cancel_order", r"\bcancel"),
    ("refund_request", r"\brefund|paise wapas|paisa wapas|money back"),
    ("payment_issue", r"\bpayment|\bupi\b|deduct|\bkat gaye|\bemi\b|cashback|netbanking|wallet|gift card|charge"),
    ("product_complaint", r"damage|toota|tuta|broken|defective|kharab|wrong item|galat|fake|quality|missing|used item|loose|alag"),
    ("delivery_delay", r"\blate\b|delay|ho gaye|abhi tak nahi aaya|not delivered|postpone|atka|nikal gayi|aaya hi nahi"),
    ("order_status", r"status|track|kahan|kab tak|dispatch|shipped|out for delivery|where is|pack ho"),
)
_NEGATIVE = re.compile(r"\[angry\]|\[sad\]|ghatiya|bakwas|fraud|kharab|toota|damaged|fake|nahi aaya|late|disappointed|mazaak|barbaad")
_POSITIVE = re.compile(r"\bthanks?\b|thank you|\bmast\b|great|badhiya|\[positive\]")
_HIGH = re.compile(r"jaldi|turant|urgent|asap|!!|fraud|court|\[urgent\]|toota|fake|kat gaye|deduct")
_LOW = re.compile(r"\?$|kya aap|kaise|kahan|please|\[please\]")


def rule_based_extraction(text: str) -> Extraction:
    """Label ``text`` with simple keyword rules (deterministic)."""
    lowered = text.lower()
    intent = next((label for label, pattern in _INTENT_RULES if re.search(pattern, lowered)), "other")
    if _NEGATIVE.search(lowered):
        sentiment = "negative"
    elif _POSITIVE.search(lowered):
        sentiment = "positive"
    else:
        sentiment = "neutral"
    if _HIGH.search(lowered):
        urgency = "high"
    elif intent == "other" or sentiment == "positive" or _LOW.search(lowered):
        urgency = "low"
    else:
        urgency = "medium"
    return Extraction(intent=intent, entities=extract_entities(text), sentiment=sentiment, urgency=urgency)


class MockBackend:
    """Fake backend with deterministic output and approximate token counts."""

    def __init__(self, simulated_latency_ms: int = 0) -> None:
        self._latency_s = simulated_latency_ms / 1000.0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Return rule-based JSON for ``request.contents``."""
        if self._latency_s:
            await asyncio.sleep(self._latency_s)
        payload = json.dumps(rule_based_extraction(request.contents).model_dump(), ensure_ascii=False)
        input_tokens = (len(request.system_instruction) + len(request.contents)) // 4
        return LLMResponse(text=payload, input_tokens=input_tokens, output_tokens=len(payload) // 4)
