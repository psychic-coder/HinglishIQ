"""UI-independent helpers used by the Streamlit app (testable without Streamlit)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.baseline import BaselineModel, BaselineNotTrainedError
from src.llm_client import LLMClient, PredictionResult
from src.pipeline import PipelinePrediction, predict
from src.prompts import PromptSpec, discover_prompts
from src.schemas import Extraction

EXAMPLE_MESSAGES: dict[str, str] = {
    "Late order + refund": "bhai mera order 3 din se nahi aaya, refund chahiye 😡",
    "Double payment": "do baar paise kat gaye ₹1,299 order OD48213377 ke liye, jldi dekho!!",
    "Damaged product": "mixer grinder toota hua aaya hai, packaging bhi phati thi 😤",
    "Polite status check": "hi, mera order OD7730091 kab tak deliver hoga? thanks 🙏",
    "Cancel request": "galti se order ho gaya, cancel kr do pls, 2 din me shaadi h",
}


@dataclass(frozen=True)
class SinglePrediction:
    """Everything the "Try it" tab displays for one message."""

    llm: PipelinePrediction
    baseline: Extraction | None

    @property
    def result(self) -> PredictionResult:
        """Shortcut to the LLM prediction result."""
        return self.llm.result


@dataclass(frozen=True)
class ResultsBundle:
    """Evaluation artifacts found in the results directory."""

    metrics_markdown: str | None
    confusion_images: list[Path] = field(default_factory=list)

    @property
    def available(self) -> bool:
        """True if ``python main.py evaluate`` has produced a report."""
        return self.metrics_markdown is not None


def list_prompt_versions(prompts_dir: Path | str) -> list[str]:
    """Sorted prompt version labels discovered in ``prompts_dir``."""
    return sorted(discover_prompts(prompts_dir))


def try_load_baseline(path: Path | str) -> BaselineModel | None:
    """Return the trained baseline, or None if it has not been trained yet."""
    try:
        return BaselineModel.load(path)
    except BaselineNotTrainedError:
        return None


def run_single_prediction(
    text: str, prompt: PromptSpec, client: LLMClient, baseline: BaselineModel | None
) -> SinglePrediction:
    """Run the LLM pipeline (and baseline, if available) on one message."""
    if not text.strip():
        raise ValueError("Please enter a message.")
    llm = predict(client, prompt, text)
    base = baseline.predict([text])[0] if baseline is not None else None
    return SinglePrediction(llm=llm, baseline=base)


def friendly_error(result: PredictionResult) -> str | None:
    """Human-readable explanation of a failed LLM call (None if it succeeded)."""
    if result.error is None:
        return None
    if result.error_kind == "rate_limit":
        return ("The Gemini free tier is rate-limiting requests (HTTP 429 / RESOURCE_EXHAUSTED). "
                "Wait a minute and retry, lower rate_limit_rpm in config.yaml, or switch to mock mode.")
    lowered = result.error.lower()
    if "api key" in lowered or "api_key" in lowered or "permission" in lowered or "401" in lowered or "403" in lowered:
        return "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file, or switch to mock mode."
    if "not found" in lowered or "404" in lowered:
        return "The configured model was not found. Confirm the free-tier model name in config/config.yaml."
    return f"The Gemini API call failed: {result.error}. You can retry or switch to mock mode."


def load_results(results_dir: Path | str) -> ResultsBundle:
    """Collect ``metrics.md`` and confusion PNGs (gracefully empty if absent)."""
    results_dir = Path(results_dir)
    metrics_path = results_dir / "metrics.md"
    markdown = metrics_path.read_text(encoding="utf-8") if metrics_path.is_file() else None
    images = sorted(results_dir.glob("confusion_*.png")) if results_dir.is_dir() else []
    return ResultsBundle(metrics_markdown=markdown, confusion_images=images)
