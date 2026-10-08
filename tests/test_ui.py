"""ui_helpers functions and an app.py smoke test (no server is launched)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from conftest import FakeBackend, make_client
from src.backends import TransientAPIError
from src.pipeline import MissingAPIKeyError, build_llm_client
from src.ui_helpers import (
    EXAMPLE_MESSAGES,
    friendly_error,
    list_prompt_versions,
    load_results,
    run_single_prediction,
    try_load_baseline,
)

ROOT = Path(__file__).resolve().parent.parent


def test_list_prompt_versions():
    assert list_prompt_versions(ROOT / "prompts") == ["v1", "v2", "v3"]


def test_list_prompt_versions_empty_dir(tmp_path: Path):
    assert list_prompt_versions(tmp_path) == []


def test_run_single_prediction_mock(tmp_config, prompt_v3):
    client = build_llm_client(tmp_config, mock=True)
    out = run_single_prediction(EXAMPLE_MESSAGES["Late order + refund"], prompt_v3, client, baseline=None)
    assert out.result.valid_json and out.result.error is None
    assert out.result.extraction.intent == "refund_request"
    assert out.baseline is None
    assert 0.0 <= out.llm.preprocessed.code_mix_ratio <= 1.0
    with pytest.raises(ValueError):
        run_single_prediction("   ", prompt_v3, client, None)


def test_missing_baseline_returns_none(tmp_path: Path):
    assert try_load_baseline(tmp_path / "nope.joblib") is None


def test_missing_results_handled(tmp_path: Path):
    bundle = load_results(tmp_path / "does-not-exist")
    assert not bundle.available and bundle.confusion_images == []


def test_results_found(tmp_path: Path):
    (tmp_path / "metrics.md").write_text("# hi", encoding="utf-8")
    (tmp_path / "confusion_llm_v1.png").write_bytes(b"png")
    bundle = load_results(tmp_path)
    assert bundle.available and bundle.confusion_images[0].name == "confusion_llm_v1.png"


def test_friendly_error_messages(prompt_v3):
    limited = make_client(FakeBackend([TransientAPIError(429, "RESOURCE_EXHAUSTED")]), max_retries=0).extract("x", prompt_v3)
    assert "rate-limiting" in friendly_error(limited)
    bad_key = make_client(FakeBackend([TransientAPIError(400, "API key not valid")])).extract("x", prompt_v3)
    assert "API key" in friendly_error(bad_key)


def test_real_mode_without_key_raises(tmp_config, monkeypatch):
    monkeypatch.setattr("src.pipeline.get_api_key", lambda: None)
    with pytest.raises(MissingAPIKeyError):
        build_llm_client(tmp_config, mock=False)


def test_app_imports_without_launching_server():
    app = importlib.import_module("app")
    assert callable(app.main) and callable(app.tab_try_it)


def test_app_renders_in_mock_mode(monkeypatch):
    """Run the script headlessly with Streamlit's AppTest (no server, no network)."""
    monkeypatch.setattr("src.pipeline.get_api_key", lambda: None)
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("MOCK MODE" in w.value for w in at.warning)
    example_box = next(s for s in at.selectbox if s.label == "Example messages")
    example_box.select("Double payment").run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert not at.json
    developer_details = next(checkbox for checkbox in at.checkbox if checkbox.label == "Developer details")
    developer_details.check().run()
    at.button[0].click().run()
    assert len(at.json) >= 1
    assert any(m.label == "Valid JSON" and m.value == "yes" for m in at.metric)
