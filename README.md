# HinglishIQ

**Code-mixed (Hinglish) e-commerce support message understanding with an LLM, benchmarked against a classical baseline.**

Customers of Indian e-commerce apps write things like:

> `bhai mera order 3 din se nahi aaya, refund chahiye 😡`

HinglishIQ turns such noisy Hindi-in-Roman-script + English messages into validated JSON:

```json
{"intent": "refund_request", "entities": [{"type": "duration", "value": "3 days"}], "sentiment": "negative", "urgency": "high"}
```

It calls the **Google Gemini API (free tier)** with versioned, few-shot prompts and structured output, validates every response with Pydantic (with one repair retry), and compares three prompt versions against a TF-IDF + logistic-regression baseline on a held-out test split. A CLI and a Streamlit demo UI sit on top of the same pipeline.

| Field | Labels |
|---|---|
| intent | `refund_request`, `order_status`, `delivery_delay`, `cancel_order`, `product_complaint`, `payment_issue`, `other` |
| entities | `order_id`, `product`, `duration`, `amount`, `date` |
| sentiment | `negative`, `neutral`, `positive` |
| urgency | `low`, `medium`, `high` |

---

## Architecture

```
                 raw Hinglish message
                          │
                          ▼
              ┌──────────────────────┐
              │  src/preprocess.py   │  NFKC, lowercase, emoji → [angry]/[positive] cues,
              │                      │  "pleaseeee"→"pleasee", slang (nhi→nahi, kr→kar),
              │                      │  IDs/amounts kept, code-mix ratio (Hindi lexicon)
              └──────────┬───────────┘
            ┌────────────┴─────────────┐
            ▼                          ▼
┌───────────────────────┐   ┌──────────────────────────────────────────────┐
│   src/baseline.py     │   │              src/llm_client.py               │
│ char_wb TF-IDF (2-5)  │   │  cache (.cache/, SHA-256 of model+temp+FULL  │
│ + LogisticRegression  │   │        prompt file hash+normalized input)    │
│ ×3 heads (intent,     │   │  → rate limiter (rpm) → semaphore (conc.)    │
│ sentiment, urgency)   │   │  → backend.generate()  ── 429/5xx? backoff+  │
│ + src/entities.py     │   │                            jitter, retry     │
│   regex entities      │   │  → Pydantic validate ── invalid? ONE repair  │
└──────────┬────────────┘   │    retry ── still invalid? fallback "other"  │
           │                │  → PredictionResult (latency, tokens, cache, │
           │                │    valid_json, retries, error)               │
           │                └──────────┬───────────────────────────────────┘
           │                  backends │ src/backends.py: GeminiBackend (google-genai,
           │                           │   response_schema=Extraction, temperature 0)
           │                           │ src/mock_llm.py: MockBackend (deterministic, offline)
           │                           │ prompts/*.yaml via src/prompts.py
           ▼                           ▼
          │                           │
        │ src/evaluate.py + src/metrics.py          │ dev for tuning, test for final numbers
        │ intent acc/F1, sentiment F1, urgency F1,  │ → results/metrics.json, metrics.md,
        │ entity P/R/F1, JSON validity, p50/p95,    │   confusion_<method>.png,
        │ tokens/sample, cache hit, API calls,      │   errors_<method>.jsonl
        │ accuracy by code-mix bin                  │
        └──────────────────────────────────────────┘
                 ▲                         ▲
          main.py (CLI)          app.py (Streamlit) → src/ui_helpers.py
```

### Repository layout

```
hinglishiq/
├── app.py                  # Streamlit UI (thin layer)
├── main.py                 # CLI: split | train-baseline | predict | evaluate
├── config/config.yaml      # all tunables (model, rate limits, paths, seeds, splits)
├── pytest.ini              # test discovery (tests/, offline)
├── prompts/                # v1_zero_shot.yaml, v2_few_shot.yaml, v3_schema_rules.yaml
├── data/                   # samples.jsonl (+ train/dev/test.jsonl after `split`)
├── src/
│   ├── config.py           # typed + validated config (Pydantic)
│   ├── schemas.py          # Extraction / Entity / LabeledSample
│   ├── preprocess.py       # normalization + code-mix ratio
│   ├── entities.py         # entity normalization + regex extractor
│   ├── prompts.py          # prompt YAML loader/validator/renderer
│   ├── backends.py         # LLM interface, Gemini backend, schema compatibility, errors
│   ├── mock_llm.py         # deterministic fake backend
│   ├── rate_limiter.py     # rpm limiter, per-event-loop primitives
│   ├── cache.py            # content-addressed disk cache
│   ├── llm_client.py       # retries, repair, fallback, telemetry
│   ├── pipeline.py         # client factory + preprocess→LLM
│   ├── baseline.py         # TF-IDF + LogisticRegression
│   ├── metrics.py          # scoring functions
│   ├── evaluate.py         # benchmark runner + reports
│   ├── split_data.py       # stratified split, few-shot pinning
│   ├── data_io.py          # JSONL I/O
│   └── ui_helpers.py       # UI logic, testable without Streamlit
├── tests/                  # pytest, offline, no key needed
└── results/                # written by `evaluate`
```

---

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Create .env and add GEMINI_API_KEY=... when using real Gemini mode.
```

### Getting a free Gemini API key

1. Go to **Google AI Studio**: <https://aistudio.google.com/app/apikey> and sign in with a Google account.
2. Click **Create API key**. No billing account is needed for the free tier.
3. Paste the key into `.env`: `GEMINI_API_KEY=...`. The key is only read from this environment variable and is never hard-coded. `.env` is git-ignored.
4. In AI Studio, check which **Flash / Flash-Lite** models are currently on the free tier and make sure `llm.model` in `config/config.yaml` matches (default `gemini-3.5-flash-lite`). Pro models are rejected by the config validator.

Everything except real-mode `predict`/`evaluate` works **without a key** via `--mock`.

---

## Usage

```bash
# 1. stratified train/dev/test split (50/20/30, seed 42; few-shot examples pinned to train)
python main.py split

# 2. train the classical baseline on train.jsonl → models/baseline.joblib
python main.py train-baseline

# 3. full benchmark offline with the deterministic fake LLM (no key, no network)
python main.py evaluate --mock

# 4. real benchmark with Gemini on the dev split (use this while iterating on prompts)
python main.py evaluate --split dev

#    final numbers on the held-out test split (run once, after prompt tuning is frozen)
python main.py evaluate --split test

#    subsets
python main.py evaluate --prompts v2,v3 --limit 20

# 5. single message (default prompt from config; --mock for offline)
python main.py predict "bhai mera order 3 din se nahi aaya, refund chahiye"
python main.py predict "paise kat gaye ₹1,299 order OD4821337 jldi dekho!!" --prompt v2 --mock

# 6. demo UI
streamlit run app.py

# 7. tests (offline)
pytest
```

`predict` prints the input, the preprocessed text, the code-mix ratio, the prediction and metadata (latency, tokens, cache hit, valid JSON, retries, error). Add `-v` before the subcommand for debug logs. The CLI accepts `--config PATH` to load an alternate YAML configuration.

### Free-tier notes

Free-tier limits (requests per minute, per day and tokens per minute) are small and change over time. HinglishIQ therefore:

- paces requests client-side (`reliability.rate_limit_rpm`, default 10/min) and caps parallelism (`reliability.max_concurrency`, default 2);
- retries 429 / `RESOURCE_EXHAUSTED` / 5xx with exponential backoff and jitter (`max_retries`, `backoff_base_seconds`, `backoff_max_seconds`);
- caches every successful response in `.cache/`, so **re-running an evaluation is free** and makes zero API calls. Editing a prompt file changes its content hash and invalidates only that prompt's cache entries.

A full dev evaluation of three prompts is 3 × 42 = 126 calls (about 13 minutes at 10 rpm). A test evaluation is 3 × 63 = 189 calls. If you hit daily quotas, use `--prompts` and `--limit`, or lower the rpm.

---

## Prompt files and ablation (v1 → v2 → v3)

Each file in `prompts/` has `version`, `description`, `changelog`, `system_prompt` and `few_shot` (list of `{input, output}`). The loader rejects files with missing fields or few-shot outputs that violate the schema. The few-shot inputs are rendered through the same preprocessing as live inputs. The JSON structure is enforced API-side with `response_mime_type="application/json"`, `temperature=0`, and a Gemini-compatible schema derived from `Extraction`. The compatibility schema removes Pydantic's `additionalProperties` keyword because the Gemini API rejects that JSON Schema field; Pydantic still validates every returned object locally.

| Version | What changed | Why |
|---|---|---|
| **v1** zero-shot | Role, task, every label defined; no examples | Measures what label definitions alone achieve. |
| **v2** few-shot | + 12 Hinglish examples covering all 7 intents and all 5 entity types, with typos, emojis and Roman-script Hindi | v1 has no anchor for entity value formats or slang. Examples teach the style and include hard pairs: a neutral `delivery_delay`, refund vs payment, praise → `other`. |
| **v3** schema + rules | + ordered decision rules (main request wins; `delivery_delay` vs `order_status`; `payment_issue` vs `refund_request` vs `cancel_order`; complaint vs refund), urgency and sentiment cue lists, entity normalization (`"3 din"→"3 days"`, `"Rs 1,299"→"₹1299"`, upper-case IDs, `"8 nov"→"8 november"`), "output ONLY the JSON" | Targets the confusable intent pairs and aligns entity strings with the exact-match metric. |

The few-shot examples are real items from `data/samples.jsonl`. `split_data.py` **pins them to the train split**, and `tests/test_data_and_eval.py` asserts that none of them appears in dev or test.

### Dev/test discipline

- **train** (50%): baseline training and the only source of few-shot examples.
- **dev** (20%): all prompt iteration and error analysis (`results/errors_*.jsonl`).
- **test** (30%): reported once, after prompts are frozen. `evaluate` refuses to score `train`.

---

## Results

The table below is filled from your own run. **No numbers are reported here until you run the evaluation.**

| Method | Intent acc | Intent macro-F1 | Sentiment F1 | Urgency F1 | Entity F1 | JSON valid | p50 ms | p95 ms | Tokens/sample |
|---|---|---|---|---|---|---|---|---|---|
| baseline | run `python main.py evaluate` to populate | | | | | | | | |
| llm_v1 | run `python main.py evaluate` to populate | | | | | | | | |
| llm_v2 | run `python main.py evaluate` to populate | | | | | | | | |
| llm_v3 | run `python main.py evaluate` to populate | | | | | | | | |

How to fill it in: `python main.py evaluate --split test` writes `results/metrics.md`, which has the same columns plus entity P/R, cache hit rate and API calls. It also has a second table of intent accuracy by code-mix bin (low < 0.50 ≤ med < 0.65 ≤ high, configurable in `evaluation:`). Copy that table here. `results/metrics.json` has the raw numbers, including entity TP/FP/FN, retries and error counts. `results/confusion_<method>.png` are the intent confusion matrices, and `results/errors_<method>.jsonl` lists every sample where a method was wrong on any field, with its preprocessed text and code-mix ratio for error analysis. A report generated with `--mock` is labelled **MOCK MODE** and must not be reported as Gemini results.

Notes on metrics: entity scores are micro P/R/F1 over exact `(type, normalized value)` matches, with normalization in `src/entities.py`. JSON validity counts responses that passed Pydantic validation, either directly or after the single repair retry. Failed calls count as invalid. Latency is wall-clock per message and includes time waiting for the rate limiter and the concurrency semaphore. Cached calls report 0 tokens.

---

## Demo UI (`streamlit run app.py`)

The app is a thin layer over `src/`. All logic lives in `src/ui_helpers.py` and the pipeline.

- **Sidebar**: prompt-version selector (discovered from `prompts/`), a **Mock mode** checkbox (on by default when `GEMINI_API_KEY` is missing), the model name (read-only), and an opt-in **Developer details** checkbox. A **"MOCK MODE: results are fake"** banner is shown whenever mock mode is on.
- **Try it**: pick one of five example messages or type your own, then click **Run**. The default view shows a plain-language request summary, customer mood, suggested priority, and detected details. Raw JSON, model telemetry, preprocessed text, and baseline internals appear only when **Developer details** is enabled. API failures and rate limits appear as a friendly error that suggests mock mode. The app never switches to mock silently.
- **Results**: renders `results/metrics.md` and every `results/confusion_*.png`. If they are missing, it shows how to run `python main.py evaluate`.
- **Prompts**: version, description, changelog, few-shot count and the full system prompt of the selected file.

---

## Engineering notes

- **Concurrency model:** orchestration is `asyncio` (rate limiter, semaphore, `gather`). The Gemini backend runs the SDK's synchronous `client.models.generate_content` in worker threads via `asyncio.to_thread`. This keeps the backend independent of any event loop, so the sync wrappers (`LLMClient.extract`, `pipeline.predict`) can call `asyncio.run` repeatedly from the CLI and from Streamlit reruns. Locks and semaphores are re-created per event loop (`LoopLocal`).
- **Token usage** is read from `response.usage_metadata` (prompt, candidate and thinking tokens). It falls back to 0 when the metadata is missing.
- **Mock mode** (`src/mock_llm.py`) implements the same backend interface with transparent keyword rules and the regex entity extractor. It uses its own model name, so its cache entries never mix with Gemini's.
- **No global mutable state:** stats, caches and limiters live on client instances. The Streamlit app holds them with `st.cache_resource`.

---

## Dataset

`data/samples.jsonl` holds **210 labelled messages (30 per intent)**.

> **This seed set is synthetic.** It was generated programmatically from hand-written Hinglish templates with an LLM's assistance, with random typos, slang spellings, emojis, order IDs (`OD…`), amounts (`₹499`, `499 rs`, `Rs 1,299`, `799/-`), durations and dates. **It is not real customer data.** The author must manually review the labels and extend the set with more diverse, ideally real (anonymised), messages before drawing conclusions.

Fields: `id`, `text`, `intent`, `entities` (canonical values), `sentiment`, `urgency`.

---

## Limitations

- **Synthetic, template-based data.** Messages built from the same template share structure across splits, so scores (especially the baseline's and the regex entities') are likely optimistic compared with real traffic. Products are the only entity type the regex baseline cannot find.
- **Small test set** (63 messages, about 9 per intent). Confidence intervals are wide: a single message moves intent accuracy by about 1.6 points. Differences of a few points between prompts are not significant.
- **Free-tier limits** cap how much can be evaluated per day, and the available models and quotas change. Results are tied to the model version in `config.yaml`.
- **Code-mix ratio** is a lexicon heuristic (~180 romanized Hindi words). Words like "me" are ambiguous between Hindi and English.
- **Labels are subjective**, especially urgency, and a single annotator wrote them.

## Future work

- Collect and annotate real (consented, anonymised) Hinglish tickets, and measure inter-annotator agreement.
- Add bootstrap confidence intervals and McNemar tests between methods.
- Try dynamic few-shot selection (retrieve the k nearest train examples per message).
- Fine-tune a small multilingual encoder (e.g. MuRIL or IndicBERT) as a stronger baseline.
- Support Devanagari and mixed-script input with transliteration.
- Add per-field confidence and route low-confidence messages to a human.

---

## Run it (from clone to first result)

```bash
git clone <your-repo-url> hinglishiq && cd hinglishiq
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Create .env and add GEMINI_API_KEY=... for real mode.
python main.py split
python main.py train-baseline
python main.py evaluate --mock
pytest
streamlit run app.py                    # mock mode is on automatically without a key
# real mode: put your key in .env (GEMINI_API_KEY=...), confirm the model name in config/config.yaml, then
python main.py predict "bhai mera order 3 din se nahi aaya, refund chahiye"
python main.py evaluate --split dev
python main.py evaluate --split test
```

---

## Project purpose

HinglishIQ is a small, reproducible benchmark and demonstration system for classifying customer-support messages written in code-mixed Hindi and English. It is designed around four practical requirements:

1. **Noisy input:** customers use Roman-script Hindi, English, abbreviations, spelling mistakes, repeated punctuation, emojis, and inconsistent casing.
2. **Structured decisions:** downstream support workflows need a stable intent, entities, sentiment, and urgency rather than an unconstrained paragraph.
3. **Reliable LLM integration:** API calls need validation, retries, rate limiting, caching, and an offline mode.
4. **Measurable improvement:** prompt changes should be compared against a classical baseline on held-out data.

The project is an experiment and reference implementation, not a production support router. The dataset is synthetic and the output should be reviewed before being used for customer-facing automation.

## End-to-end lifecycle

For one message, the application follows this sequence:

1. The caller supplies raw text through the CLI or Streamlit UI.
2. `src.preprocess` applies Unicode normalization, lowercase conversion, emoji cue replacement, elongation and punctuation cleanup, slang normalization, order-ID restoration, whitespace normalization, tokenization, and code-mix estimation.
3. The pipeline sends the normalized text to either `GeminiBackend` or `MockBackend`.
4. `LLMClient` checks the disk cache, applies the rate limiter and concurrency semaphore, and calls the backend.
5. Transient failures are retried with exponential backoff and jitter. Non-retryable errors fail fast.
6. The response is parsed as JSON and validated against the Pydantic `Extraction` model.
7. Invalid JSON receives exactly one repair request. If repair also fails, the safe fallback is `other`, no entities, neutral sentiment, and low urgency.
8. A `PredictionResult` records the extraction and telemetry.
9. The CLI serializes the result, while the UI translates it into a readable support summary.

The evaluation path uses the same preprocessing and prediction pipeline, then compares predictions with gold labels and writes metrics, confusion matrices, and error records.

## Output contract

Every successful prediction has this shape:

```json
{
  "intent": "refund_request",
  "entities": [
    {"type": "duration", "value": "3 days"}
  ],
  "sentiment": "negative",
  "urgency": "high"
}
```

### Intent labels

| Label | Meaning |
|---|---|
| `refund_request` | The customer asks for money back or a refund. |
| `order_status` | The customer asks where an order is or when it will arrive. |
| `delivery_delay` | The customer reports that an expected delivery is late. |
| `cancel_order` | The customer wants an order cancelled. |
| `product_complaint` | The customer reports a damaged, defective, missing, or poor-quality product. |
| `payment_issue` | The customer reports a failed, duplicate, missing, or incorrect payment or charge. |
| `other` | No supported intent is sufficiently clear. |

Prompt rules make the distinction between similar labels explicit. For example, a request for a refund takes precedence over a delivery complaint when both appear, and a payment charge problem is distinct from a refund request.

### Entity labels

| Type | Canonical examples | Extracted by regex baseline |
|---|---|---|
| `order_id` | `OD48213377` | Yes |
| `product` | `mixer grinder` | No; the LLM or mock rules can provide it |
| `duration` | `3 days`, `2 weeks` | Yes |
| `amount` | `₹1299` | Yes |
| `date` | `8 november` | Yes |

Entity scoring compares exact `(type, normalized value)` pairs. The displayed value is expected to be canonical, not necessarily a verbatim substring of the message.

### Sentiment and urgency

Sentiment is one of `negative`, `neutral`, or `positive`. Urgency is one of `low`, `medium`, or `high`. These are classification labels, not calibrated probabilities. The system intentionally does not return confidence scores because the current models do not provide a validated confidence estimate.

## Preprocessing reference

`preprocess(text)` returns `PreprocessResult` with `original`, normalized `text`, token tuple, and `code_mix_ratio`.

| Operation | Example |
|---|---|
| Unicode NFKC | Normalizes compatible Unicode forms. |
| Lowercase | `PLEASE help` becomes `please help`. |
| Emoji cues | Angry emojis become `[angry]`; positive emojis become `[positive]`; prayer emojis become `[please]`. |
| Elongation cleanup | `pleaseeee` becomes `pleasee`. |
| Punctuation cleanup | Long runs such as `!!!!` are shortened to `!!`. |
| Slang normalization | `nhi`, `kr`, `jldi`, and `plz` become canonical forms such as `nahi`, `kar`, `jaldi`, and `please`. |
| ID restoration | `od 48213377` becomes `OD48213377`. |
| Whitespace normalization | Repeated spaces are collapsed. |

The code-mix ratio is the fraction of alphabetic tokens found in the built-in Romanized-Hindi lexicon. It is a diagnostic feature and evaluation grouping, not a language detector. Ambiguous words and unseen slang can make it imperfect.

## Entity normalization reference

`src/entities.py` canonicalizes values before exact-match scoring:

| Input forms | Canonical value |
|---|---|
| `OD 48213377`, `#ord-48213377` | `OD48213377` |
| `Rs. 1,299`, `1299 rupees`, `₹1,299` | `₹1299` |
| `3 din`, `3 days` | `3 days` |
| `2 hafta`, `2 weeks` | `2 weeks` |
| `8 nov`, `8th November` | `8 november` |

The regex extractor scans order IDs, amounts, durations, and dates. It de-duplicates identical normalized entities. Product extraction is intentionally left to the LLM/mock path because product names are open-ended.

## Baseline model

The baseline is deliberately simple and inspectable:

- Input: preprocessed message text.
- Features: character-boundary TF-IDF n-grams, default range 2 through 5.
- Classifier: one balanced `LogisticRegression` pipeline per head: intent, sentiment, and urgency.
- Entities: deterministic regex extraction from the raw/preprocessed text.
- Persistence: the three scikit-learn pipelines are saved with `joblib`.

Train it with:

```bash
python main.py split
python main.py train-baseline
```

The saved file defaults to `models/baseline.joblib`. The UI can still run without it, but it will show that the baseline has not been trained. Evaluation requires the saved baseline.

## LLM client behavior

`LLMClient` is the reliability boundary around either backend.

### Cache

Successful responses are stored under `reliability.cache_dir`, default `.cache/`. The SHA-256 cache key includes:

- model name;
- temperature;
- the full prompt file content hash; and
- normalized input text.

Changing a prompt file invalidates only entries for that prompt. Failed calls are not cached. Cache hits report zero input and output tokens and zero backend API calls for that request.

### Retries and repair

The client retries HTTP 429, `RESOURCE_EXHAUSTED`, HTTP 5xx, timeouts, and connection failures. Backoff is exponential with additive random jitter and a configured maximum. Invalid response JSON is a separate validation path and receives one repair attempt; it is not retried indefinitely.

Bad requests, invalid keys, permission errors, and unknown models are classified as non-retryable API errors. The UI maps common errors to plain-language guidance and never silently changes real mode to mock mode.

### Concurrency

`aextract_many` runs calls with `asyncio.gather`, bounded by a semaphore and a requests-per-minute limiter. The Gemini SDK call itself is synchronous and runs in a worker thread with `asyncio.to_thread`. Loop-local locks and semaphores allow the same client design to work across repeated CLI and Streamlit event loops.

### Gemini structured output compatibility

The Pydantic model is used for local validation. Before it is sent to Gemini, `src.backends._gemini_response_schema()` recursively removes `additionalProperties` from the generated JSON Schema. Gemini's API rejects that keyword even though it is valid JSON Schema and is emitted by Pydantic for `extra="forbid"` models.

## Configuration reference

The default file is `config/config.yaml`. `src.config.load_config` validates the complete document with Pydantic, rejects unknown top-level fields, resolves relative paths against the project root, and rejects model names containing `pro` because the project targets free-tier Flash models.

| Path | Default | Purpose |
|---|---:|---|
| `llm.model` | `gemini-3.5-flash-lite` | Gemini model ID for real mode. Verify current availability before use. |
| `llm.temperature` | `0.0` | Deterministic structured generation. |
| `llm.max_output_tokens` | `512` | Maximum generated output. |
| `llm.thinking_budget` | `null` | Optional Gemini thinking-token budget. |
| `reliability.rate_limit_rpm` | `10` | Local requests-per-minute pacing. `0` disables pacing. |
| `reliability.max_concurrency` | `2` | Maximum in-flight requests. |
| `reliability.max_retries` | `4` | Retries after the initial call. |
| `reliability.backoff_base_seconds` | `2.0` | Initial retry delay plus jitter. |
| `reliability.backoff_max_seconds` | `60.0` | Maximum retry delay. |
| `reliability.cache_dir` | `.cache` | Disk cache location. |
| `paths.data_dir` | `data` | Generated train/dev/test directory. |
| `paths.samples_file` | `data/samples.jsonl` | Source dataset. |
| `paths.prompts_dir` | `prompts` | Prompt YAML directory. |
| `paths.results_dir` | `results` | Evaluation output directory. |
| `paths.baseline_model` | `models/baseline.joblib` | Saved baseline path. |
| `split.train/dev/test` | `0.5/0.2/0.3` | Stratified split proportions. |
| `random_seed` | `42` | Split, classifier, and mock determinism. |
| `default_prompt_version` | `v3` | Prompt selected by default. |
| `evaluation.codemix_low_max` | `0.50` | Upper bound for the low bin. |
| `evaluation.codemix_med_max` | `0.65` | Upper bound for the medium bin. |
| `baseline.ngram_min/max` | `2/5` | Character n-gram range. |
| `baseline.C` | `5.0` | Logistic regression regularization. |
| `baseline.max_iter` | `2000` | Classifier iteration limit. |
| `mock.model_name` | `mock-hinglish-rules` | Model label used in mock telemetry/cache keys. |
| `mock.rate_limit_rpm` | `0` | Mock-mode request pacing. |
| `mock.simulated_latency_ms` | `5` | Deterministic mock delay. |

For a different configuration:

```bash
python main.py --config path/to/config.yaml predict "order kahan hai" --mock
```

## Data format

### Source and split JSONL

Each line in `data/samples.jsonl` and its generated split files is a JSON object:

```json
{
  "id": "sample-001",
  "text": "bhai mera order 3 din se nahi aaya",
  "intent": "delivery_delay",
  "entities": [{"type": "duration", "value": "3 days"}],
  "sentiment": "neutral",
  "urgency": "medium"
}
```

`id` is a stable sample identifier, `text` is the raw message, and the remaining fields are the gold `Extraction`. The loader validates every line with Pydantic and reports malformed JSON or schema violations as `DataError`.

### Prompt YAML

Each prompt file has this conceptual shape:

```yaml
version: v3
description: Schema and decision rules
changelog:
  - Added ordered intent rules
system_prompt: |
  ...
few_shot:
  - input: "message"
    output:
      intent: other
      entities: []
      sentiment: neutral
      urgency: low
```

Few-shot outputs must satisfy the same extraction schema as live responses. Few-shot inputs are preprocessed before rendering so training examples and live requests use the same normalization path.

## CLI reference

### `split`

Creates stratified `train.jsonl`, `dev.jsonl`, and `test.jsonl` under `paths.data_dir`. The configured proportions and seed are used. The examples referenced by prompt few-shot sets are pinned to the train split to prevent prompt leakage into dev or test.

### `train-baseline`

Loads the train split, fits the three classifier heads, and writes the joblib model. Run this after changing the split or baseline configuration.

### `predict`

Analyzes one raw message. Options:

- `--prompt VERSION`: select a discovered prompt, such as `v2`.
- `--mock`: use the deterministic offline backend.
- `--config PATH`: use a different configuration file.
- `-v` or `--verbose`: enable debug logging.

Exit code is `0` for a successful call, `1` for configuration/input setup errors, and `2` when the LLM call fails and the output is a fallback.

### `evaluate`

Runs the baseline and selected prompt versions on `dev` or `test`, then writes all report artifacts. Options:

- `--mock`: produce an offline report and label it MOCK MODE.
- `--split dev|test`: choose the evaluation split; train is intentionally rejected.
- `--prompts v1,v2`: evaluate only selected prompt versions.
- `--limit N`: evaluate only the first N samples after loading the split.

Use dev for prompt iteration. Use test only after prompts and rules are frozen.

## Evaluation methodology

For each method, the evaluator computes:

- intent accuracy and macro-F1;
- sentiment macro-F1;
- urgency macro-F1;
- micro entity precision, recall, and F1 over normalized `(type, value)` pairs;
- JSON validity for LLM methods;
- latency p50 and p95;
- average tokens per sample;
- cache hit rate, API calls, retries, and errors; and
- intent accuracy in low, medium, and high code-mix bins.

The baseline receives JSON validity of 100% and zero token/API counts because it is not an LLM. LLM JSON validity includes responses that become valid after the one repair request. A failed request is invalid and is represented by the safe fallback in error artifacts.

## Generated artifacts

After `evaluate`, `results/` contains:

| File pattern | Contents |
|---|---|
| `metrics.json` | Machine-readable report, model, mode, timestamp, and all metrics. |
| `metrics.md` | Human-readable Markdown report. |
| `confusion_<method>.png` | Intent confusion matrix for each method. |
| `errors_<method>.jsonl` | Samples with mismatched fields, invalid JSON, or API errors. |

Error records include the sample ID, raw and preprocessed text, code-mix ratio, gold extraction, predicted extraction, validity, and error message. This makes prompt debugging possible without re-running the whole benchmark.

## Streamlit UI reference

Start the application with:

```bash
streamlit run app.py
```

The app runs in mock mode by default when `GEMINI_API_KEY` is absent. It does not silently fall back from real mode to mock mode.

### Try it workflow

1. Choose a prompt version.
2. Leave mock mode enabled for offline exploration, or disable it after configuring a valid key and model.
3. Choose an example message or enter a custom message.
4. Select **Run**.
5. Read the plain-language request summary, mood, priority, and detected details.
6. Enable **Developer details** only when raw JSON, latency, token counts, preprocessing, or baseline diagnostics are needed.

The UI's readable labels are presentation-only. The underlying contract remains the exact `Extraction` schema used by the CLI and evaluator.

### Results tab

The Results tab displays the latest `results/metrics.md` and all confusion-matrix PNGs. If no report exists, it explains which evaluation command to run.

### Prompts tab

The Prompts tab displays the selected prompt's description, changelog, few-shot count, and full system prompt. This is intended for prompt auditing and iteration.

## Testing and quality checks

Tests are offline and do not require a Gemini key or network access:

```bash
.venv/bin/python -m pytest -q
```

The suite covers:

- preprocessing, slang, emoji cues, and code-mix ratios;
- entity extraction and canonicalization;
- Pydantic schema and prompt validation;
- split reproducibility and few-shot leakage prevention;
- baseline training, persistence, and prediction;
- retry, rate-limit, repair, fallback, cache, and telemetry behavior;
- evaluation metrics and generated artifacts; and
- Streamlit import and AppTest rendering, including mock mode and developer-details opt-in behavior.

Before opening a pull request, run the full suite and at least one mock prediction. For changes to evaluation or prompts, run a limited mock evaluation first:

```bash
python main.py evaluate --mock --limit 10
```

## Troubleshooting

### `ModuleNotFoundError: streamlit`

Activate the project environment and install requirements:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS Homebrew Python may be externally managed. Use a virtual environment; do not install project packages into the system interpreter.

### Gemini says the model was not found

Model availability and free-tier access change. Check the current model ID in Google AI Studio and update `llm.model`. The current default is `gemini-3.5-flash-lite`. Older 2.5 Flash-Lite accounts may receive a 404 because Google restricts some 2.5 model access for new users.

### Gemini rejects the response schema

Use the current `src.backends.GeminiBackend`, which sends the sanitized schema generated by `_gemini_response_schema`. Do not revert to passing `Extraction.model_json_schema()` unchanged because Gemini rejects `additionalProperties`.

### API key errors or permission errors

Confirm that `.env` contains `GEMINI_API_KEY=...`, that the process is running from the project directory, and that the key belongs to a region/account allowed to use the selected model. Use `--mock` while diagnosing the rest of the application.

### Rate limits or slow evaluations

Use `--mock`, reduce `--limit`, evaluate one prompt with `--prompts v3`, lower concurrency, or rely on the cache for repeated requests. Do not treat a mock report as a Gemini benchmark.

### Baseline is missing

Run `python main.py split` followed by `python main.py train-baseline`. The UI can still demonstrate LLM/mock predictions without the baseline.

### Stale results or cache entries

Results are files, not live views of the current code. Re-run evaluation after changing prompts or model configuration. To invalidate all local cached LLM responses, remove `.cache/`; cached files are ignored by Git.

## Security and privacy

- Never commit `.env`, API keys, generated caches, or customer data.
- The API key is read from `GEMINI_API_KEY`; it is not hard-coded or included in prompts.
- The seed dataset is synthetic and contains no intended customer records.
- Real messages may contain order IDs, amounts, or personal information. Add redaction, access control, retention policy, and provider review before using this project with production data.
- The generated error files copy raw and preprocessed messages, so treat `results/` as potentially sensitive when evaluating real data.

## Extending the project

### Add a prompt version

1. Copy an existing YAML file under `prompts/`.
2. Change `version`, description, changelog, system prompt, and examples.
3. Ensure every few-shot output passes the `Extraction` schema.
4. Run `pytest`, then `python main.py evaluate --mock --prompts <version> --limit 10`.
5. Tune on dev and freeze before evaluating test.

### Add an intent or entity type

Update the relevant `Literal` in `src/schemas.py`, label-space constants, prompt instructions and examples, mock rules, baseline/evaluation assumptions, and tests. For a new regex entity, update both normalization and extraction in `src/entities.py` and add exact-match test cases.

### Add a backend

Implement the `LLMBackend` protocol's async `generate(LLMRequest) -> LLMResponse` method. Keep transport concerns in the backend; retries, caching, validation, and telemetry belong in `LLMClient`. Add a deterministic fake or scripted backend for tests instead of making tests call a live service.

### Add a UI field

Keep business logic in `src/ui_helpers.py` or the pipeline and keep `app.py` as a thin rendering layer. Add an AppTest assertion for the visible behavior and preserve the default non-technical view unless the field is intended for developers.

## Reproducibility checklist

For a result that another developer can reproduce:

1. Record the Git revision and Python version.
2. Record the exact `config/config.yaml`, prompt files, and model ID.
3. Run `python main.py split` with the configured seed.
4. Run `python main.py train-baseline`.
5. Run the same `evaluate` command, including split, prompt list, limit, and mock/real mode.
6. Preserve `results/metrics.json`, the Markdown report, and error files.
7. For real Gemini runs, record the run date because model availability, quotas, and service behavior change.

## License and contribution note

No license file is currently included in this repository. Add an explicit license before distributing the project beyond its intended private or educational use. Contributions should include focused tests and should avoid committing `.env`, `.cache/`, model artifacts, or generated result files unless the change specifically documents them.
