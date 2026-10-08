"""Pydantic v2 schemas: the LLM output contract and labelled samples.

``Extraction`` is passed directly to Gemini as ``response_schema`` and is also
used to validate whatever text comes back, so it intentionally has no default
values (structured-output schemas work best with all fields required).
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

Intent = Literal[
    "refund_request",
    "order_status",
    "delivery_delay",
    "cancel_order",
    "product_complaint",
    "payment_issue",
    "other",
]
EntityType = Literal["order_id", "product", "duration", "amount", "date"]
Sentiment = Literal["negative", "neutral", "positive"]
Urgency = Literal["low", "medium", "high"]

INTENTS: tuple[str, ...] = get_args(Intent)
ENTITY_TYPES: tuple[str, ...] = get_args(EntityType)
SENTIMENTS: tuple[str, ...] = get_args(Sentiment)
URGENCIES: tuple[str, ...] = get_args(Urgency)

#: Label spaces for the three classification heads, keyed by field name.
LABEL_SPACES: dict[str, tuple[str, ...]] = {
    "intent": INTENTS,
    "sentiment": SENTIMENTS,
    "urgency": URGENCIES,
}


class Entity(BaseModel):
    """A single extracted entity."""

    model_config = ConfigDict(extra="forbid")

    type: EntityType
    value: str = Field(min_length=1)


class Extraction(BaseModel):
    """Structured understanding of one customer-support message."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    entities: list[Entity]
    sentiment: Sentiment
    urgency: Urgency

    @classmethod
    def fallback(cls) -> "Extraction":
        """Safe default used when the LLM output cannot be validated."""
        return cls(intent="other", entities=[], sentiment="neutral", urgency="low")


class LabeledSample(BaseModel):
    """One gold-labelled message from ``data/*.jsonl``."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str = Field(min_length=1)
    intent: Intent
    entities: list[Entity]
    sentiment: Sentiment
    urgency: Urgency

    def gold(self) -> Extraction:
        """Return the gold labels as an ``Extraction``."""
        return Extraction(
            intent=self.intent,
            entities=self.entities,
            sentiment=self.sentiment,
            urgency=self.urgency,
        )
