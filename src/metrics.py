"""Metric computations shared by evaluation and tests."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.entities import normalize_entities
from src.schemas import LABEL_SPACES, Entity, Extraction

CODEMIX_BINS: tuple[str, ...] = ("low", "med", "high")


def entity_prf(gold: Sequence[Sequence[Entity]], pred: Sequence[Sequence[Entity]]) -> dict[str, float]:
    """Micro precision/recall/F1 over exact (type, normalized value) matches.

    Duplicates are handled as multisets, so predicting the same entity twice
    yields one true positive and one false positive.
    """
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")
    tp = fp = fn = 0
    for gold_entities, pred_entities in zip(gold, pred):
        g = Counter(normalize_entities(list(gold_entities)))
        p = Counter(normalize_entities(list(pred_entities)))
        overlap = sum((g & p).values())
        tp += overlap
        fp += sum(p.values()) - overlap
        fn += sum(g.values()) - overlap
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def head_scores(gold: Sequence[Extraction], pred: Sequence[Extraction]) -> dict[str, float]:
    """Intent accuracy + macro-F1 for intent, sentiment and urgency."""
    scores: dict[str, float] = {}
    for head in LABEL_SPACES:
        y_true = [getattr(g, head) for g in gold]
        y_pred = [getattr(p, head) for p in pred]
        # macro over labels present in gold or predictions (absent labels would add spurious zeros)
        scores[f"{head}_macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if head == "intent":
            scores["intent_accuracy"] = float(accuracy_score(y_true, y_pred))
    return scores


def codemix_bin(ratio: float, low_max: float, med_max: float) -> str:
    """Bucket a code-mix ratio into low / med / high."""
    if ratio < low_max:
        return "low"
    if ratio < med_max:
        return "med"
    return "high"


def accuracy_by_bin(
    gold: Sequence[Extraction], pred: Sequence[Extraction], bins: Sequence[str]
) -> dict[str, dict[str, float | None]]:
    """Intent accuracy and sample count per code-mix bin (accuracy None if empty)."""
    result: dict[str, dict[str, float | None]] = {}
    for name in CODEMIX_BINS:
        idx = [i for i, b in enumerate(bins) if b == name]
        correct = sum(gold[i].intent == pred[i].intent for i in idx)
        result[name] = {"n": len(idx), "intent_accuracy": correct / len(idx) if idx else None}
    return result


def latency_percentiles(latencies_ms: Sequence[float]) -> dict[str, float]:
    """p50 / p95 latency in milliseconds (0 for empty input)."""
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0}
    arr = np.asarray(latencies_ms, dtype=float)
    return {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95))}


def mismatched_fields(gold: Extraction, pred: Extraction) -> list[str]:
    """Names of fields where ``pred`` disagrees with ``gold``."""
    fields = [h for h in LABEL_SPACES if getattr(gold, h) != getattr(pred, h)]
    if normalize_entities(gold.entities) != normalize_entities(pred.entities):
        fields.append("entities")
    return fields
