"""Schemas, preprocessing, entities/metrics, rate limiter and prompt loader."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.entities import extract_entities, normalize_entity_value
from src.metrics import codemix_bin, entity_prf, head_scores, mismatched_fields
from src.preprocess import preprocess
from src.prompts import PromptError, discover_prompts, load_prompt, render_system_instruction
from src.rate_limiter import RateLimiter
from src.schemas import Entity, Extraction

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------- schemas
def test_schema_accepts_valid():
    e = Extraction.model_validate({"intent": "order_status", "entities": [], "sentiment": "neutral", "urgency": "low"})
    assert e.intent == "order_status"


@pytest.mark.parametrize("bad", [
    {"intent": "angry", "entities": [], "sentiment": "neutral", "urgency": "low"},
    {"intent": "other", "entities": [], "sentiment": "furious", "urgency": "low"},
    {"intent": "other", "entities": [], "sentiment": "neutral", "urgency": "critical"},
    {"intent": "other", "entities": [{"type": "phone", "value": "123"}], "sentiment": "neutral", "urgency": "low"},
    {"intent": "other", "entities": [], "sentiment": "neutral"},
    {"intent": "other", "entities": [], "sentiment": "neutral", "urgency": "low", "extra": 1},
])
def test_schema_rejects_bad_labels(bad):
    with pytest.raises(ValidationError):
        Extraction.model_validate(bad)


# ------------------------------------------------------------------ preprocess
def test_preprocess_slang_elongation_and_ids():
    r = preprocess("Bhai order od123456 NHI aaya pleaseeee jldi kro!!!!")
    assert "OD123456" in r.text
    assert "nahi" in r.text and "jaldi karo" in r.text
    assert "pleasee" in r.text and "pleaseee" not in r.text
    assert "!!!!" not in r.text


def test_preprocess_keeps_numbers_and_amounts():
    r = preprocess("₹1000 kat gaye, 3000 rs bhi")
    assert "₹1000" in r.text and "3000" in r.text


def test_preprocess_emoji_cues():
    assert "[angry]" in preprocess("kya bakwas hai 😡").text
    assert "[positive]" in preprocess("mast product 👍").text
    assert "🦄" not in preprocess("hello 🦄").text


def test_code_mix_ratio_bounds():
    assert preprocess("where is my order please").code_mix_ratio < 0.2
    assert preprocess("mera order abhi tak nahi aaya bhai").code_mix_ratio > 0.6
    assert preprocess("12345 !!").code_mix_ratio == 0.0


def test_codemix_bins():
    assert [codemix_bin(x, 0.5, 0.65) for x in (0.1, 0.5, 0.64, 0.65, 0.9)] == ["low", "med", "med", "high", "high"]


# -------------------------------------------------------------------- entities
@pytest.mark.parametrize("etype,raw,canon", [
    ("duration", "3 din", "3 days"), ("duration", "2 hafte", "2 weeks"), ("duration", "5 dino", "5 days"),
    ("amount", "Rs. 1,299", "₹1299"), ("amount", "499 rs", "₹499"), ("amount", "₹ 799/-", "₹799"),
    ("order_id", "#od123456", "OD123456"), ("date", "8th Nov", "8 november"), ("product", " Nike  Shoes", "nike shoes"),
])
def test_entity_normalization(etype, raw, canon):
    assert normalize_entity_value(etype, raw) == canon


def test_regex_extractor():
    found = {(e.type, e.value) for e in extract_entities("order OD5566778 ka ₹1,499 refund 3 din se pending, 12 march ko diya")}
    assert found == {("order_id", "OD5566778"), ("amount", "₹1499"), ("duration", "3 days"), ("date", "12 march")}


def test_entity_f1_hand_made():
    gold = [[Entity(type="duration", value="3 days"), Entity(type="amount", value="₹499")], [], [Entity(type="product", value="kurti")]]
    pred = [[Entity(type="duration", value="3 din"), Entity(type="amount", value="₹500")], [Entity(type="order_id", value="OD1")], []]
    m = entity_prf(gold, pred)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 2, 2)
    assert m["precision"] == pytest.approx(1 / 3) and m["recall"] == pytest.approx(1 / 3)
    assert m["f1"] == pytest.approx(1 / 3)


def test_entity_f1_duplicates_and_empty():
    e = Entity(type="product", value="saree")
    assert entity_prf([[e]], [[e, e]])["fp"] == 1
    assert entity_prf([[]], [[]])["f1"] == 0.0


def test_head_scores_and_mismatch():
    g = Extraction(intent="cancel_order", entities=[], sentiment="neutral", urgency="low")
    p = Extraction(intent="cancel_order", entities=[Entity(type="amount", value="₹1")], sentiment="negative", urgency="low")
    assert head_scores([g], [p])["intent_accuracy"] == 1.0
    assert mismatched_fields(g, p) == ["sentiment", "entities"]


# ---------------------------------------------------------------- rate limiter
def test_rate_limiter_spacing():
    now = [100.0]
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(rpm=30, clock=lambda: now[0], sleep=fake_sleep)  # 2 s apart

    async def run() -> list[float]:
        return [await limiter.acquire() for _ in range(4)]

    waits = asyncio.run(run())
    assert waits == [0.0, 2.0, 2.0, 2.0]
    assert slept == [2.0, 2.0, 2.0]


def test_rate_limiter_concurrent_slots_are_spaced():
    now = [0.0]
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    limiter = RateLimiter(rpm=60, clock=lambda: now[0], sleep=fake_sleep)

    async def run() -> None:
        await asyncio.gather(*(limiter.acquire() for _ in range(3)))

    asyncio.run(run())
    assert sorted(slept) == [1.0, 2.0]


def test_rate_limiter_disabled_and_reusable_across_loops():
    limiter = RateLimiter(rpm=0)
    assert asyncio.run(limiter.acquire()) == 0.0
    assert asyncio.run(limiter.acquire()) == 0.0
    with pytest.raises(ValueError):
        RateLimiter(rpm=-1)


# ------------------------------------------------------------------ prompts
def test_all_prompt_files_load():
    versions = discover_prompts(ROOT / "prompts")
    assert set(versions) == {"v1", "v2", "v3"}
    assert len(load_prompt(versions["v1"]).few_shot) == 0
    assert 10 <= len(load_prompt(versions["v2"]).few_shot) <= 12
    v3 = load_prompt(versions["v3"])
    rendered = render_system_instruction(v3)
    assert "EXAMPLES" in rendered and "DECISION RULES" in rendered


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "p.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_prompt_loader_missing_field(tmp_path: Path):
    path = _write(tmp_path, "version: vx\ndescription: d\nsystem_prompt: s\nfew_shot: []\n")
    with pytest.raises(PromptError, match="changelog"):
        load_prompt(path)


def test_prompt_loader_bad_few_shot(tmp_path: Path):
    body = ("version: vx\ndescription: d\nchangelog: c\nsystem_prompt: s\nfew_shot:\n"
            "  - input: hi\n    output: {intent: nope, entities: [], sentiment: neutral, urgency: low}\n")
    with pytest.raises(PromptError, match="few_shot\\[0\\]"):
        load_prompt(_write(tmp_path, body))


def test_prompt_loader_not_mapping_and_missing_file(tmp_path: Path):
    with pytest.raises(PromptError):
        load_prompt(_write(tmp_path, "- just a list\n"))
    with pytest.raises(PromptError, match="not found"):
        load_prompt(tmp_path / "nope.yaml")


def test_prompt_content_hash_changes_with_content(tmp_path: Path):
    base = "version: vx\ndescription: d\nchangelog: c\nsystem_prompt: s\nfew_shot: []\n"
    h1 = load_prompt(_write(tmp_path, base)).content_hash
    h2 = load_prompt(_write(tmp_path, base.replace("system_prompt: s", "system_prompt: s2"))).content_hash
    assert h1 != h2
