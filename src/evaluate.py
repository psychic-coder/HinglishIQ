"""Benchmark the baseline and every prompt version on a data split."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from src.baseline import BaselineModel  # noqa: E402
from src.config import AppConfig  # noqa: E402
from src.data_io import load_samples  # noqa: E402
from src.llm_client import LLMClient, PredictionResult  # noqa: E402
from src.metrics import (  # noqa: E402
    accuracy_by_bin,
    codemix_bin,
    entity_prf,
    head_scores,
    latency_percentiles,
    mismatched_fields,
)
from src.pipeline import build_llm_client, predict_many  # noqa: E402
from src.preprocess import PreprocessResult, preprocess  # noqa: E402
from src.prompts import discover_prompts, load_prompt  # noqa: E402
from src.schemas import INTENTS, Extraction, LabeledSample  # noqa: E402

logger = logging.getLogger(__name__)

BASELINE_NAME = "baseline"


@dataclass
class MethodRun:
    """Predictions and telemetry for one method on one split."""

    name: str
    predictions: list[Extraction]
    latencies_ms: list[float]
    telemetry: list[PredictionResult] | None = None
    api_calls: int = 0


def run_baseline(model: BaselineModel, samples: Sequence[LabeledSample]) -> MethodRun:
    """Predict each sample with the baseline, timing every call."""
    predictions: list[Extraction] = []
    latencies: list[float] = []
    for sample in samples:
        start = time.perf_counter()
        predictions.append(model.predict([sample.text])[0])
        latencies.append((time.perf_counter() - start) * 1000.0)
    return MethodRun(BASELINE_NAME, predictions, latencies)


def run_prompt(client: LLMClient, prompt_path: Path, samples: Sequence[LabeledSample]) -> MethodRun:
    """Run one prompt version over all samples via the LLM pipeline."""
    prompt = load_prompt(prompt_path)
    client.reset_stats()
    outputs = predict_many(client, prompt, [s.text for s in samples])
    telemetry = [o.result for o in outputs]
    return MethodRun(
        name=f"llm_{prompt.version}",
        predictions=[t.extraction for t in telemetry],
        latencies_ms=[t.latency_ms for t in telemetry],
        telemetry=telemetry,
        api_calls=client.stats.api_calls,
    )


def score_run(
    run: MethodRun, samples: Sequence[LabeledSample], bins: Sequence[str]
) -> dict[str, Any]:
    """Compute every metric for one method."""
    gold = [s.gold() for s in samples]
    n = len(samples)
    metrics: dict[str, Any] = head_scores(gold, run.predictions)
    entity = entity_prf([g.entities for g in gold], [p.entities for p in run.predictions])
    metrics.update({f"entity_{k}": v for k, v in entity.items()})
    metrics.update(latency_percentiles(run.latencies_ms))
    if run.telemetry is None:
        metrics.update({"json_validity": 1.0, "tokens_per_sample": 0.0, "cache_hit_rate": None,
                        "api_calls": 0, "retries": 0, "errors": 0})
    else:
        t = run.telemetry
        metrics.update({
            "json_validity": sum(r.valid_json for r in t) / n if n else 0.0,
            "tokens_per_sample": sum(r.input_tokens + r.output_tokens for r in t) / n if n else 0.0,
            "cache_hit_rate": sum(r.cache_hit for r in t) / n if n else 0.0,
            "api_calls": run.api_calls,
            "retries": sum(r.retries for r in t),
            "errors": sum(r.error is not None for r in t),
        })
    metrics["by_codemix"] = accuracy_by_bin(gold, run.predictions, bins)
    metrics["n"] = n
    return metrics


def _fmt(value: Any, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if pct:
        return f"{100 * value:.1f}%"
    if isinstance(value, float):
        return f"{value:.3f}" if value < 10 else f"{value:.0f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    """Render the metrics report as Markdown tables."""
    lines = [f"# HinglishIQ results ({report['split']} split, n={report['n_samples']})", ""]
    if report["mode"] == "mock":
        lines += ["> **MOCK MODE: LLM rows are produced by a deterministic fake client, not Gemini.**", ""]
    lines += [f"Model: `{report['model']}` | generated {report['generated_at']}", ""]
    header = ("| Method | Intent acc | Intent F1 | Sentiment F1 | Urgency F1 | Entity P | Entity R "
              "| Entity F1 | JSON valid | p50 ms | p95 ms | Tokens/sample | Cache hit | API calls |")
    lines += [header, "|" + "---|" * 14]
    for name, m in report["methods"].items():
        lines.append(
            f"| {name} | {_fmt(m['intent_accuracy'])} | {_fmt(m['intent_macro_f1'])} "
            f"| {_fmt(m['sentiment_macro_f1'])} | {_fmt(m['urgency_macro_f1'])} "
            f"| {_fmt(m['entity_precision'])} | {_fmt(m['entity_recall'])} | {_fmt(m['entity_f1'])} "
            f"| {_fmt(m['json_validity'], pct=True)} | {m['p50_ms']:.1f} | {m['p95_ms']:.1f} "
            f"| {m['tokens_per_sample']:.0f} | {_fmt(m['cache_hit_rate'], pct=True)} | {m['api_calls']} |"
        )
    lines += ["", "## Intent accuracy by code-mix ratio", "",
              "| Method | low | med | high |", "|---|---|---|---|"]
    for name, m in report["methods"].items():
        cells = [f"{_fmt(m['by_codemix'][b]['intent_accuracy'])} (n={m['by_codemix'][b]['n']})"
                 for b in ("low", "med", "high")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_confusion(run: MethodRun, samples: Sequence[LabeledSample], path: Path) -> None:
    """Save an intent confusion-matrix PNG."""
    labels = list(INTENTS)
    matrix = confusion_matrix([s.intent for s in samples], [p.intent for p in run.predictions], labels=labels)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("Gold intent")
    ax.set_title(f"Intent confusion: {run.name}")
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j]:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > threshold else "#1f2937", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_errors(
    run: MethodRun, samples: Sequence[LabeledSample], prepped: Sequence[PreprocessResult], path: Path
) -> int:
    """Write mispredicted samples as JSONL; return the count."""
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for i, (sample, pred) in enumerate(zip(samples, run.predictions)):
            fields = mismatched_fields(sample.gold(), pred)
            telemetry = run.telemetry[i] if run.telemetry else None
            if not fields and (telemetry is None or telemetry.valid_json):
                continue
            record = {
                "id": sample.id,
                "text": sample.text,
                "preprocessed": prepped[i].text,
                "code_mix_ratio": prepped[i].code_mix_ratio,
                "mismatched": fields,
                "gold": sample.gold().model_dump(),
                "pred": pred.model_dump(),
                "valid_json": telemetry.valid_json if telemetry else True,
                "error": telemetry.error if telemetry else None,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def evaluate(
    config: AppConfig,
    split: str = "dev",
    prompt_versions: Sequence[str] | None = None,
    mock: bool = False,
    limit: int | None = None,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """Run baseline + prompt versions on ``split`` and write all result files.

    Returns the report dict that is also saved to ``results/metrics.json``.
    """
    if split not in ("dev", "test"):
        raise ValueError("split must be 'dev' or 'test' (never evaluate on train)")
    samples = load_samples(config.split_file(split))
    if limit:
        samples = samples[:limit]
    available = discover_prompts(config.paths.prompts_dir)
    versions = list(prompt_versions) if prompt_versions else sorted(available)
    unknown = [v for v in versions if v not in available]
    if unknown:
        raise ValueError(f"Unknown prompt version(s): {unknown}. Available: {sorted(available)}")

    baseline = BaselineModel.load(config.paths.baseline_model)
    client = client or build_llm_client(config, mock=mock)
    prepped = [preprocess(s.text) for s in samples]
    ev = config.evaluation
    bins = [codemix_bin(p.code_mix_ratio, ev.codemix_low_max, ev.codemix_med_max) for p in prepped]

    runs = [run_baseline(baseline, samples)]
    for version in versions:
        logger.info("Evaluating prompt %s on %d %s samples", version, len(samples), split)
        runs.append(run_prompt(client, available[version], samples))

    report: dict[str, Any] = {
        "split": split,
        "mode": "mock" if client.is_mock else "real",
        "model": client.model,
        "n_samples": len(samples),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "methods": {run.name: score_run(run, samples, bins) for run in runs},
    }
    results_dir = config.paths.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (results_dir / "metrics.md").write_text(render_markdown(report), encoding="utf-8")
    for run in runs:
        save_confusion(run, samples, results_dir / f"confusion_{run.name}.png")
        save_errors(run, samples, prepped, results_dir / f"errors_{run.name}.jsonl")
    return report
