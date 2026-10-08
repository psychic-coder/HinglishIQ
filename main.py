"""HinglishIQ command-line interface.

Usage:
    python main.py split
    python main.py train-baseline
    python main.py predict "bhai mera order 3 din se nahi aaya" [--prompt v2] [--mock]
    python main.py evaluate [--mock] [--split dev|test] [--prompts v1,v2,v3] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

from src.baseline import BaselineModel, BaselineNotTrainedError
from src.config import DEFAULT_CONFIG_PATH, AppConfig, ConfigError, load_config
from src.data_io import DataError, load_samples
from src.evaluate import evaluate, render_markdown
from src.pipeline import MissingAPIKeyError, build_llm_client, predict
from src.prompts import PromptError, resolve_prompt
from src.split_data import run_split

logger = logging.getLogger("hinglishiq")


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_split(config: AppConfig, _args: argparse.Namespace) -> int:
    """Create stratified train/dev/test files."""
    sizes = run_split(config)
    _print_json({"split_sizes": sizes, "data_dir": str(config.paths.data_dir)})
    return 0


def cmd_train_baseline(config: AppConfig, _args: argparse.Namespace) -> int:
    """Train the TF-IDF + LogisticRegression baseline on the train split."""
    samples = load_samples(config.split_file("train"))
    model = BaselineModel.train(samples, config.baseline, config.random_seed)
    model.save(config.paths.baseline_model)
    _print_json({"trained_on": len(samples), "saved_to": str(config.paths.baseline_model)})
    return 0


def cmd_predict(config: AppConfig, args: argparse.Namespace) -> int:
    """Run the full pipeline on one message and pretty-print the JSON."""
    prompt = resolve_prompt(config.paths.prompts_dir, args.prompt or config.default_prompt_version)
    client = build_llm_client(config, mock=args.mock)
    output = predict(client, prompt, args.text)
    result = output.result
    payload = {
        "input": output.original,
        "preprocessed": output.preprocessed.text,
        "code_mix_ratio": output.preprocessed.code_mix_ratio,
        "prediction": result.extraction.model_dump(),
        "meta": {k: v for k, v in result.to_dict().items() if k != "extraction"}
        | {"prompt_version": prompt.version, "model": client.model, "mode": "mock" if client.is_mock else "real"},
    }
    _print_json(payload)
    if result.error:
        print(f"\nWARNING: LLM call failed ({result.error}); prediction is the fallback.", file=sys.stderr)
        return 2
    return 0


def cmd_evaluate(config: AppConfig, args: argparse.Namespace) -> int:
    """Benchmark baseline + prompt versions and write results/."""
    versions = [v.strip() for v in args.prompts.split(",") if v.strip()] if args.prompts else None
    report = evaluate(config, split=args.split, prompt_versions=versions, mock=args.mock, limit=args.limit)
    print(render_markdown(report))
    print(f"Saved metrics.json, metrics.md, confusion_*.png and errors_*.jsonl to {config.paths.results_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse CLI."""
    parser = argparse.ArgumentParser(prog="hinglishiq", description="Hinglish support-message understanding")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("split", help="stratified train/dev/test split").set_defaults(func=cmd_split)
    sub.add_parser("train-baseline", help="train the classical baseline").set_defaults(func=cmd_train_baseline)

    p_predict = sub.add_parser("predict", help="analyse one message")
    p_predict.add_argument("text", help="Hinglish message (quote it)")
    p_predict.add_argument("--prompt", default=None, help="prompt version, e.g. v2 (default from config)")
    p_predict.add_argument("--mock", action="store_true", help="use the deterministic fake LLM")
    p_predict.set_defaults(func=cmd_predict)

    p_eval = sub.add_parser("evaluate", help="benchmark baseline and prompts")
    p_eval.add_argument("--mock", action="store_true", help="use the deterministic fake LLM")
    p_eval.add_argument("--split", choices=("dev", "test"), default="dev")
    p_eval.add_argument("--prompts", default=None, help="comma-separated versions, e.g. v1,v2,v3 (default: all)")
    p_eval.add_argument("--limit", type=int, default=None, help="only the first N samples")
    p_eval.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        config = load_config(args.config)
        return args.func(config, args)
    except (ConfigError, PromptError, DataError, BaselineNotTrainedError, MissingAPIKeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
