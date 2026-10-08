"""Classical baseline: char n-gram TF-IDF + logistic regression per head."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import BaselineSettings
from src.entities import extract_entities
from src.preprocess import preprocess
from src.schemas import LABEL_SPACES, Extraction, LabeledSample

logger = logging.getLogger(__name__)

HEADS: tuple[str, ...] = tuple(LABEL_SPACES)  # intent, sentiment, urgency


class BaselineNotTrainedError(FileNotFoundError):
    """Raised when the saved baseline model file does not exist."""


def _make_pipeline(settings: BaselineSettings, seed: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(settings.ngram_min, settings.ngram_max),
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", LogisticRegression(
            C=settings.C,
            max_iter=settings.max_iter,
            class_weight="balanced",
            random_state=seed,
        )),
    ])


class BaselineModel:
    """Three independent classifiers plus regex entities."""

    def __init__(self, pipelines: dict[str, Pipeline]) -> None:
        missing = set(HEADS) - set(pipelines)
        if missing:
            raise ValueError(f"Baseline is missing heads: {sorted(missing)}")
        self.pipelines = pipelines

    @classmethod
    def train(cls, samples: Sequence[LabeledSample], settings: BaselineSettings, seed: int) -> "BaselineModel":
        """Fit one pipeline per head on preprocessed text."""
        texts = [preprocess(s.text).text for s in samples]
        pipelines = {}
        for head in HEADS:
            pipeline = _make_pipeline(settings, seed)
            pipeline.fit(texts, [getattr(s, head) for s in samples])
            pipelines[head] = pipeline
        logger.info("Trained baseline on %d samples", len(samples))
        return cls(pipelines)

    def predict(self, texts: Sequence[str]) -> list[Extraction]:
        """Predict ``Extraction`` objects for raw messages."""
        prepped = [preprocess(t).text for t in texts]
        heads = {head: self.pipelines[head].predict(prepped) for head in HEADS}
        return [
            Extraction(
                intent=heads["intent"][i],
                entities=extract_entities(text),
                sentiment=heads["sentiment"][i],
                urgency=heads["urgency"][i],
            )
            for i, text in enumerate(prepped)
        ]

    def save(self, path: Path | str) -> None:
        """Persist with joblib (parent directories are created)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipelines, path)

    @classmethod
    def load(cls, path: Path | str) -> "BaselineModel":
        """Load a model saved by :meth:`save`."""
        path = Path(path)
        if not path.is_file():
            raise BaselineNotTrainedError(
                f"Baseline model not found at {path}. Run `python main.py train-baseline` first."
            )
        return cls(joblib.load(path))
