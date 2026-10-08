"""Split leakage, config validation, baseline and end-to-end mock evaluation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from src.baseline import BaselineModel, BaselineNotTrainedError
from src.config import ConfigError, load_config
from src.data_io import load_samples, match_key
from src.evaluate import evaluate
from src.split_data import few_shot_inputs, run_split, split_samples

ROOT = Path(__file__).resolve().parent.parent


def test_dataset_is_large_and_balanced():
    samples = load_samples(ROOT / "data" / "samples.jsonl")
    assert len(samples) >= 150
    counts = Counter(s.intent for s in samples)
    assert len(counts) == 7 and min(counts.values()) >= 20
    assert len({s.id for s in samples}) == len(samples)


def test_split_has_no_few_shot_leakage(tmp_config):
    sizes = run_split(tmp_config)
    assert sum(sizes.values()) == len(load_samples(tmp_config.paths.samples_file))
    pinned = few_shot_inputs(tmp_config.paths.prompts_dir)
    assert pinned, "prompts should contain few-shot examples"
    train_keys = {match_key(s.text) for s in load_samples(tmp_config.split_file("train"))}
    for split in ("dev", "test"):
        keys = {match_key(s.text) for s in load_samples(tmp_config.split_file(split))}
        assert not (keys & pinned), f"few-shot example leaked into {split}"
        assert not (keys & train_keys), f"{split} overlaps train"
    assert pinned <= train_keys, "every few-shot example must come from train"


def test_committed_splits_have_no_leakage():
    """Also guard the split files shipped in data/ (if they exist)."""
    pinned = few_shot_inputs(ROOT / "prompts")
    for split in ("dev", "test"):
        path = ROOT / "data" / f"{split}.jsonl"
        if path.exists():
            assert not ({match_key(s.text) for s in load_samples(path)} & pinned)


def test_split_is_deterministic_and_stratified():
    samples = load_samples(ROOT / "data" / "samples.jsonl")
    a = split_samples(samples, (0.5, 0.2, 0.3), 42)
    b = split_samples(samples, (0.5, 0.2, 0.3), 42)
    assert [s.id for s in a["test"]] == [s.id for s in b["test"]]
    test_counts = Counter(s.intent for s in a["test"])
    assert max(test_counts.values()) - min(test_counts.values()) <= 1


def test_config_validation(tmp_path: Path):
    raw = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    raw["split"]["train"] = 0.9
    bad = tmp_path / "c.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="sum to 1"):
        load_config(bad)
    raw = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    raw["llm"]["model"] = "gemini-2.5-pro"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="free-tier"):
        load_config(bad)
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_baseline_train_save_load(tmp_config):
    run_split(tmp_config)
    with pytest.raises(BaselineNotTrainedError):
        BaselineModel.load(tmp_config.paths.baseline_model)
    model = BaselineModel.train(load_samples(tmp_config.split_file("train")), tmp_config.baseline, 42)
    model.save(tmp_config.paths.baseline_model)
    loaded = BaselineModel.load(tmp_config.paths.baseline_model)
    pred = loaded.predict(["order OD123456 cancel karna hai"])[0]
    assert pred.intent == "cancel_order"
    assert ("order_id", "OD123456") in {(e.type, e.value) for e in pred.entities}


def test_evaluate_end_to_end_mock(tmp_config):
    run_split(tmp_config)
    BaselineModel.train(load_samples(tmp_config.split_file("train")), tmp_config.baseline, 42).save(
        tmp_config.paths.baseline_model
    )
    report = evaluate(tmp_config, split="dev", prompt_versions=["v1", "v3"], mock=True, limit=10)
    assert report["mode"] == "mock" and report["n_samples"] == 10
    assert set(report["methods"]) == {"baseline", "llm_v1", "llm_v3"}
    assert report["methods"]["llm_v3"]["json_validity"] == 1.0
    assert report["methods"]["llm_v3"]["api_calls"] == 10
    results = tmp_config.paths.results_dir
    for name in ("metrics.json", "metrics.md", "confusion_baseline.png", "confusion_llm_v3.png", "errors_llm_v1.jsonl"):
        assert (results / name).exists(), name
    assert json.loads((results / "metrics.json").read_text())["split"] == "dev"
    # second run is fully cached: no API calls
    again = evaluate(tmp_config, split="dev", prompt_versions=["v3"], mock=True, limit=10)
    assert again["methods"]["llm_v3"]["api_calls"] == 0
    assert again["methods"]["llm_v3"]["cache_hit_rate"] == 1.0


def test_evaluate_rejects_train_split(tmp_config):
    with pytest.raises(ValueError, match="dev"):
        evaluate(tmp_config, split="train", mock=True)
