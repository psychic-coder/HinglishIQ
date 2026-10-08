"""Stratified train/dev/test split that keeps few-shot examples in train."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import AppConfig
from src.data_io import load_samples, match_key, write_samples
from src.prompts import discover_prompts, load_prompt
from src.schemas import LabeledSample

logger = logging.getLogger(__name__)

SPLITS: tuple[str, ...] = ("train", "dev", "test")


def few_shot_inputs(prompts_dir: Path | str) -> set[str]:
    """Return match keys of every few-shot input across all prompt files."""
    keys: set[str] = set()
    for path in discover_prompts(prompts_dir).values():
        keys.update(match_key(ex.input) for ex in load_prompt(path).few_shot)
    return keys


def _pin_to_train(
    splits: dict[str, list[LabeledSample]], pinned: set[str]
) -> dict[str, list[LabeledSample]]:
    """Swap pinned samples out of dev/test with same-intent train samples."""
    train = splits["train"]
    for name in ("dev", "test"):
        for index, sample in enumerate(splits[name]):
            if match_key(sample.text) not in pinned:
                continue
            swap_index = next(
                (i for i, cand in enumerate(train)
                 if cand.intent == sample.intent and match_key(cand.text) not in pinned),
                None,
            )
            if swap_index is None:
                raise ValueError(f"Cannot pin few-shot sample {sample.id} to train: no swap candidate")
            splits[name][index], train[swap_index] = train[swap_index], sample
    return splits


def split_samples(
    samples: Sequence[LabeledSample],
    ratios: tuple[float, float, float],
    seed: int,
    pinned_texts: Iterable[str] = (),
) -> dict[str, list[LabeledSample]]:
    """Stratify by intent into train/dev/test, forcing ``pinned_texts`` into train."""
    train_ratio, dev_ratio, test_ratio = ratios
    labels = [s.intent for s in samples]
    train, rest = train_test_split(
        list(samples), train_size=train_ratio, stratify=labels, random_state=seed
    )
    dev, test = train_test_split(
        rest,
        train_size=dev_ratio / (dev_ratio + test_ratio),
        stratify=[s.intent for s in rest],
        random_state=seed,
    )
    pinned = {match_key(t) for t in pinned_texts}
    missing = pinned - {match_key(s.text) for s in samples}
    if missing:
        logger.warning("%d few-shot inputs are not in the dataset (cannot be leaked)", len(missing))
    return _pin_to_train({"train": train, "dev": dev, "test": test}, pinned)


def run_split(config: AppConfig) -> dict[str, int]:
    """Split ``samples_file`` and write train/dev/test JSONL files; return sizes."""
    samples = load_samples(config.paths.samples_file)
    splits = split_samples(
        samples,
        (config.split.train, config.split.dev, config.split.test),
        config.random_seed,
        few_shot_inputs(config.paths.prompts_dir),
    )
    sizes = {name: write_samples(config.split_file(name), splits[name]) for name in SPLITS}
    logger.info("Wrote splits: %s", sizes)
    return sizes
