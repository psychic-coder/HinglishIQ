"""HinglishIQ Streamlit demo - a thin UI over the existing pipeline.

Run with:  streamlit run app.py
All logic lives in src/; this file only lays out widgets and calls helpers.
"""

from __future__ import annotations

import streamlit as st

from src.baseline import BaselineModel
from src.config import AppConfig, load_config
from src.llm_client import LLMClient
from src.pipeline import MissingAPIKeyError, build_llm_client, get_api_key
from src.prompts import PromptSpec, resolve_prompt
from src.ui_helpers import (
    EXAMPLE_MESSAGES,
    SinglePrediction,
    friendly_error,
    list_prompt_versions,
    load_results,
    run_single_prediction,
    try_load_baseline,
)


@st.cache_resource
def get_config() -> AppConfig:
    """Load the validated config once per server process."""
    return load_config()


@st.cache_resource
def get_client(mock: bool) -> LLMClient:
    """One LLM client per mode (keeps cache, rate limiter and stats alive)."""
    return build_llm_client(get_config(), mock=mock)


@st.cache_resource
def get_baseline() -> BaselineModel | None:
    """Trained baseline, or None if `python main.py train-baseline` was not run."""
    return try_load_baseline(get_config().paths.baseline_model)


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


INTENT_LABELS = {
    "refund_request": ("Refund requested", "The customer is asking for money back."),
    "order_status": ("Order status", "The customer wants to know where the order is."),
    "delivery_delay": ("Delivery is late", "The customer says the delivery has not arrived on time."),
    "cancel_order": ("Cancel an order", "The customer wants to cancel an order."),
    "product_complaint": ("Product problem", "The customer reported a problem with a product."),
    "payment_issue": ("Payment problem", "The customer reported a problem with a payment or charge."),
    "other": ("General support request", "The message does not match a specific support category."),
}
ENTITY_LABELS = {
    "order_id": "Order number",
    "product": "Product",
    "duration": "Time mentioned",
    "amount": "Amount",
    "date": "Date",
}
SENTIMENT_LABELS = {"negative": "Upset or frustrated", "neutral": "Neutral", "positive": "Positive"}
URGENCY_LABELS = {"high": "High priority", "medium": "Medium priority", "low": "Low priority"}


def render_sidebar(config: AppConfig) -> tuple[PromptSpec, bool, bool]:
    """Prompt selector, mock toggle, developer details and model info."""
    st.sidebar.header("Settings")
    versions = list_prompt_versions(config.paths.prompts_dir)
    default = versions.index(config.default_prompt_version) if config.default_prompt_version in versions else 0
    version = st.sidebar.selectbox("Prompt version", versions, index=default)
    has_key = get_api_key() is not None
    mock = st.sidebar.checkbox("Mock mode (no API calls)", value=not has_key, key="mock_mode")
    if not has_key:
        st.sidebar.caption("GEMINI_API_KEY not found - add it to .env to use real mode.")
    st.sidebar.text_input("Model", value=config.mock.model_name if mock else config.llm.model, disabled=True)
    developer_details = st.sidebar.checkbox(
        "Developer details", value=False, help="Show raw JSON, model diagnostics and baseline output."
    )
    return resolve_prompt(config.paths.prompts_dir, version), mock, developer_details


def render_prediction(prediction: SinglePrediction, show_developer_details: bool = False) -> None:
    """Show a human-readable prediction with technical details tucked away."""
    result = prediction.result
    intent_label, intent_explanation = INTENT_LABELS.get(
        result.extraction.intent, INTENT_LABELS["other"]
    )

    st.subheader("What we understood")
    st.success(f"**{intent_label}**  \n{intent_explanation}")
    summary_cols = st.columns(2)
    summary_cols[0].metric("Customer mood", SENTIMENT_LABELS[result.extraction.sentiment])
    summary_cols[1].metric("Suggested priority", URGENCY_LABELS[result.extraction.urgency])

    st.markdown("**Details mentioned**")
    if result.extraction.entities:
        for entity in result.extraction.entities:
            label = ENTITY_LABELS.get(entity.type, entity.type.replace("_", " ").title())
            st.markdown(f"- **{label}:** {entity.value}")
    else:
        st.caption("No specific order number, product, amount, date, or duration was detected.")

    if show_developer_details:
        with st.expander("Technical details", expanded=True):
            st.subheader("Structured output")
            st.json(result.extraction.model_dump())
            cols = st.columns(6)
            cols[0].metric("Latency", f"{result.latency_ms:.0f} ms")
            cols[1].metric("Tokens in / out", f"{result.input_tokens} / {result.output_tokens}")
            cols[2].metric("Cache hit", _yes_no(result.cache_hit))
            cols[3].metric("Valid JSON", _yes_no(result.valid_json))
            cols[4].metric("Retries", result.retries)
            cols[5].metric("Code-mix ratio", f"{prediction.llm.preprocessed.code_mix_ratio:.2f}")
            st.caption("Preprocessed text sent to the model:")
            st.code(prediction.llm.preprocessed.text, language=None)

        with st.expander("Baseline comparison"):
            if prediction.baseline is None:
                st.info("Baseline not trained yet. Run `python main.py train-baseline`.")
            else:
                baseline_label, _ = INTENT_LABELS.get(prediction.baseline.intent, INTENT_LABELS["other"])
                st.write(f"The classical baseline classified this as **{baseline_label}**.")
                st.json(prediction.baseline.model_dump())


def tab_try_it(prompt: PromptSpec, mock: bool, show_developer_details: bool = False) -> None:
    """Interactive single-message prediction."""
    choice = st.selectbox("Example messages", ["(write your own)", *EXAMPLE_MESSAGES])
    default_text = EXAMPLE_MESSAGES.get(choice, "")
    text = st.text_area("Hinglish message", value=default_text, height=100, key=f"text_{choice}")
    if not st.button("Run", type="primary"):
        return
    if not text.strip():
        st.warning("Please enter a message.")
        return
    try:
        client = get_client(mock)
    except MissingAPIKeyError as exc:
        st.error(str(exc))
        st.info("Tick **Mock mode** in the sidebar to try the pipeline without a key.")
        return
    with st.spinner("Analysing..."):
        prediction = run_single_prediction(text, prompt, client, get_baseline())
    message = friendly_error(prediction.result)
    if message:
        st.error(message)
        st.info("The output below is the safe fallback, not a real prediction. "
                "Tick **Mock mode** in the sidebar to keep exploring offline.")
    render_prediction(prediction, show_developer_details)


def tab_results(config: AppConfig) -> None:
    """Show the latest evaluation report and confusion matrices."""
    bundle = load_results(config.paths.results_dir)
    if not bundle.available:
        st.info("No results yet. Run `python main.py evaluate --mock` (or without --mock for Gemini).")
        return
    st.markdown(bundle.metrics_markdown)
    for image in bundle.confusion_images:
        st.image(str(image), caption=image.stem.replace("confusion_", ""))


def tab_prompts(prompt: PromptSpec) -> None:
    """Show metadata of the selected prompt file."""
    st.subheader(f"Prompt {prompt.version}")
    st.write(prompt.description)
    st.metric("Few-shot examples", len(prompt.few_shot))
    st.markdown("**Changelog**")
    st.markdown("\n".join(f"- {item}" for item in prompt.changelog))
    with st.expander("System prompt"):
        st.code(prompt.system_prompt, language=None)
    st.caption(f"File: {prompt.path.name} | content hash {prompt.content_hash[:12]}")


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="HinglishIQ", layout="wide")
    st.title("HinglishIQ")
    st.caption("Understand what the customer needs, how urgent it is, and what details they mentioned.")
    config = get_config()
    prompt, mock, show_developer_details = render_sidebar(config)
    if mock:
        st.warning("MOCK MODE: results are fake (deterministic rules, no Gemini call).")
    try_tab, results_tab, prompts_tab = st.tabs(["Try it", "Results", "Prompts"])
    with try_tab:
        tab_try_it(prompt, mock, show_developer_details)
    with results_tab:
        tab_results(config)
    with prompts_tab:
        tab_prompts(prompt)


if __name__ == "__main__":
    main()
