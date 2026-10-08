"""Entity value normalization and a regex entity extractor.

The normalizer defines the canonical form used for exact-match entity scoring
(e.g. "3 din" -> "3 days", "Rs. 1,299" -> "₹1299", "#od123456" -> "OD123456").
The regex extractor is used by the classical baseline (and the mock LLM), so
entity F1 is comparable across methods.
"""

from __future__ import annotations

import re

from src.schemas import Entity

_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_MONTH_PATTERN = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"

_DURATION_UNITS: dict[str, str] = {
    "din": "days", "dino": "days", "dinon": "days", "day": "days", "days": "days",
    "hafte": "weeks", "hafta": "weeks", "week": "weeks", "weeks": "weeks",
    "ghante": "hours", "ghanta": "hours", "hour": "hours", "hours": "hours", "hrs": "hours",
    "mahine": "months", "mahina": "months", "month": "months", "months": "months",
}
_UNIT_PATTERN = "|".join(sorted(_DURATION_UNITS, key=len, reverse=True))

ORDER_ID_RE = re.compile(r"#?\b((?:od|ord)\s?-?\d{5,12})\b", re.IGNORECASE)
AMOUNT_RE = re.compile(
    r"(?:₹|\brs\.?|\binr)\s?(\d[\d,]*)(?:\.\d+)?"
    r"|\b(\d[\d,]*)\s?(?:rs\b|rupees\b|rupaye\b|rupay\b|/-|₹)",
    re.IGNORECASE,
)
DURATION_RE = re.compile(rf"\b(\d+)\s?({_UNIT_PATTERN})\b", re.IGNORECASE)
DATE_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s({_MONTH_PATTERN})\b", re.IGNORECASE
)


def _digits(value: str) -> str:
    return re.sub(r"[^\d]", "", value)


def _normalize_amount(value: str) -> str:
    match = re.search(r"\d[\d,]*", value)
    if not match:
        return value.strip().lower()
    return f"₹{int(_digits(match.group(0)))}"


def _normalize_duration(value: str) -> str:
    match = DURATION_RE.search(value)
    if not match:
        return " ".join(value.lower().split())
    return f"{int(match.group(1))} {_DURATION_UNITS[match.group(2).lower()]}"


def _normalize_date(value: str) -> str:
    match = DATE_RE.search(value)
    if not match:
        return " ".join(value.lower().split())
    prefix = match.group(2).lower()[:3]
    month = next(m for m in _MONTHS if m.startswith(prefix))
    return f"{int(match.group(1))} {month}"


def _normalize_order_id(value: str) -> str:
    return re.sub(r"[\s#\-]", "", value).upper()


def normalize_entity_value(entity_type: str, value: str) -> str:
    """Return the canonical form of an entity value for exact-match scoring."""
    if entity_type == "amount":
        return _normalize_amount(value)
    if entity_type == "duration":
        return _normalize_duration(value)
    if entity_type == "date":
        return _normalize_date(value)
    if entity_type == "order_id":
        return _normalize_order_id(value)
    return " ".join(value.lower().split())


def normalize_entities(entities: list[Entity]) -> list[tuple[str, str]]:
    """Map entities to sorted ``(type, canonical value)`` tuples."""
    return sorted((e.type, normalize_entity_value(e.type, e.value)) for e in entities)


def extract_entities(text: str) -> list[Entity]:
    """Regex-extract order IDs, amounts, durations and dates (canonical values)."""
    found: list[Entity] = []
    seen: set[tuple[str, str]] = set()

    def add(entity_type: str, raw: str) -> None:
        key = (entity_type, normalize_entity_value(entity_type, raw))
        if key not in seen:
            seen.add(key)
            found.append(Entity(type=entity_type, value=key[1]))  # type: ignore[arg-type]

    for match in ORDER_ID_RE.finditer(text):
        add("order_id", match.group(1))
    for match in AMOUNT_RE.finditer(text):
        add("amount", match.group(1) or match.group(2))
    for match in DURATION_RE.finditer(text):
        add("duration", match.group(0))
    for match in DATE_RE.finditer(text):
        add("date", match.group(0))
    return found
