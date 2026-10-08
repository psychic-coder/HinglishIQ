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
