# AI Job Search Assistant API

A production-like backend platform for AI-assisted job applications. Built incrementally to demonstrate backend engineering skills: API design, testing, configuration management, and eventually AI workflow orchestration.

---

## Milestone 0 — Project Foundation

This milestone sets up the FastAPI project skeleton with health checking, configuration, testing, and linting. No AI logic, database, or external services yet.

**What's included:**
- FastAPI app with `/health` endpoint
- Pydantic-based settings loaded from environment variables
- Pytest test suite with two health check tests
- Ruff for linting and formatting

---

## Local Setup

```bash
# Clone the repo
git clone <repo-url>
cd ai-job-search-assistant-api

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment file
cp .env.example .env
```

---

## Run the API

```bash
uvicorn app.main:app --reload
```

Visit: `http://localhost:8000/health`  
Docs: `http://localhost:8000/docs`

---

## Run Tests

```bash
pytest
```

---

## Run Ruff

```bash
# Lint
ruff check .

# Format check
ruff format --check .

# Auto-fix
ruff check --fix .
ruff format .
```

---

---

## Milestone 1 — Application Tailoring Contract

Defines the API contract for the core AI workflow: tailoring a resume and generating application materials for a specific job. The tailoring response is **mocked and deterministic** — no LLM is called yet.

**What's included:**
- `POST /api/v1/applications/tailor` endpoint
- Pydantic request/response schemas for the full tailoring workflow
- Deterministic mock service that returns realistic structured output
- Validation: empty `master_resume` or `job_description` returns 422
- 8 new tests covering success path and validation errors

**New endpoint:**

```
POST /api/v1/applications/tailor
```

**Example request body:**

```json
{
  "master_resume": "Software engineer with 5 years of Python experience...",
  "job_description": "Looking for a backend engineer to build scalable APIs...",
  "company_info": "A fast-growing fintech startup.",
  "user_preferences": "Prefer to emphasize system design experience."
}
```

**Example curl:**

```bash
curl -X POST http://localhost:8000/api/v1/applications/tailor \
  -H "Content-Type: application/json" \
  -d '{
    "master_resume": "Software engineer with 5 years of Python experience in distributed systems.",
    "job_description": "Looking for a backend engineer to build scalable APIs using FastAPI."
  }'
```

> Note: The tailoring response is mocked and deterministic. Real LLM integration (Gemini / OpenAI) will replace the mock service in a future milestone.

---

---

## Milestone 2 — LLM Provider Abstraction

Introduces a clean abstraction layer between the service logic and the LLM vendor. The app ships with a **mock provider as the default** — no API key required. Gemini can be switched on with a single environment variable.

**What's included:**
- `app/llm/base.py` — `LLMProvider` abstract base class with a single `generate_text` method
- `app/llm/mock.py` — `MockLLMProvider` for tests and local development (deterministic, no network calls)
- `app/llm/gemini.py` — `GeminiLLMProvider` backed by `google-genai` SDK
- `app/llm/factory.py` — `get_llm_provider()` reads `LLM_PROVIDER` from config and returns the right instance
- `app/prompts/tailoring.py` — `build_tailoring_prompt()` assembles a structured prompt from the request
- Updated `app/services/application_tailoring.py` — calls the prompt builder and LLM provider; embeds provider output in the response
- 13 new tests covering factory, mock provider, prompt builder, and endpoint integration

**New environment variables:**

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock` or `gemini` |
| `GEMINI_API_KEY` | _(empty)_ | Required only when `LLM_PROVIDER=gemini` |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model name |

**Running in mock mode (default — no setup needed):**

```bash
uvicorn app.main:app --reload
```

**Running with Gemini:**

```bash
# 1. Install the SDK
pip install google-genai

# 2. Set env vars in .env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here

# 3. Start the server
uvicorn app.main:app --reload
```

> Gemini is entirely optional. All tests run against the mock provider and do not require an API key.

---

---

## Milestone 3 — LLM Error Handling & Fallback

Adds production-style resilience so provider failures (503 high demand, 429 rate limit, network errors) never crash the API. The endpoint always returns a valid `ApplicationTailorResponse`.

**What's included:**
- `app/llm/exceptions.py` — `LLMProviderError` base and `LLMProviderUnavailableError` subclass
- `app/llm/gemini.py` — all runtime SDK exceptions are now wrapped in `LLMProviderUnavailableError`
- `app/services/application_tailoring.py` — `_generate_with_fallback()` helper: tries the configured provider, falls back to `MockLLMProvider` on `LLMProviderError`
- Response signals degraded mode: `tailored_summary` and `fit_gap_analysis` include `"Fallback mode used"` when fallback occurred
- 10 new tests covering exception hierarchy, Gemini error wrapping, fallback logic, and endpoint resilience

**Fallback behavior:**

| Scenario | Provider called | Response |
|---|---|---|
| `LLM_PROVIDER=mock` | `MockLLMProvider` | Normal mock output |
| Gemini succeeds | `GeminiLLMProvider` | Real LLM output |
| Gemini raises `LLMProviderError` | Falls back to `MockLLMProvider` | `"Fallback mode used"` visible in response |
| Programming error (e.g. `TypeError`) | Not caught | 500 — intentional |

**What is intentionally not included:**
- Retries or exponential backoff (planned for a future milestone)
- A dedicated `is_fallback` field in the response schema (planned alongside response metadata)
- Tests that call the real Gemini API — all tests use monkeypatching and remain fully offline

---

---

## Milestone 4 — Structured LLM Output

Moves from "embed raw LLM text into the response" to a proper **parse → validate → map** pipeline. The LLM is now instructed to return valid JSON, which is parsed and schema-validated before being mapped into the API response.

**Why structured outputs matter for backend AI systems:**
Raw LLM text is unpredictable. A production backend needs a contract: if the AI doesn't produce the shape you expect, the system should detect it and respond safely rather than returning garbage or crashing.

**What's included:**
- `app/schemas/llm_output.py` — `TailoringLLMOutput` Pydantic model: the internal schema the LLM targets
- `app/llm/parsing.py` — `parse_tailoring_response()`: parses JSON, validates against `TailoringLLMOutput`, raises `LLMOutputParsingError` on failure
- `app/llm/exceptions.py` — `LLMOutputParsingError` added (separate from `LLMProviderError`)
- `app/llm/mock.py` — now returns deterministic valid JSON matching the schema
- `app/prompts/tailoring.py` — prompt updated to instruct the LLM to return JSON only (no markdown, no prose)
- `app/services/application_tailoring.py` — `_get_llm_output()` replaces old helper; maps parsed output directly into the response
- 15 new tests covering parser happy/error paths, mock JSON validity, fallback on parse failure

**Parsing fallback behavior:**

| Scenario | Outcome |
|---|---|
| Provider returns valid JSON | Parsed, validated, mapped to response |
| Provider returns malformed JSON | `LLMOutputParsingError` → fallback to mock |
| Provider unavailable (`LLMProviderError`) | Fallback to mock (unchanged from M3) |
| Mock provider output is invalid | Exception raised — this is a code bug, not a runtime condition |

**What is intentionally not included:**
- Gemini-native structured output mode (e.g. `response_schema` parameter) — prompt-level JSON instruction is simpler and provider-agnostic
- JSON repair / partial parsing — if the LLM returns malformed output, we fall back rather than guess
- Tests that call the real Gemini API — all tests use monkeypatching and remain fully offline

---

---

## Milestone 5 — Application Persistence

Adds a database layer so every tailoring run (inputs + AI outputs + metadata) is persisted. SQLite is the default for local development; PostgreSQL is supported by changing one environment variable.

**What's included:**
- `app/db/base.py` — SQLAlchemy `DeclarativeBase`
- `app/db/session.py` — engine, `SessionLocal`, `get_db()` FastAPI dependency
- `app/models/application.py` — `ApplicationTailoringRun` ORM model
- `app/repositories/application_runs.py` — `create_application_tailoring_run()` and `get_application_tailoring_run()`
- `app/schemas/application.py` — `ApplicationTailoringRunResponse` read schema (does not expose raw resume/JD)
- `app/services/application_tailoring.py` — now accepts `db: Session`, tracks `provider_used` / `fallback_used`, saves run after generation
- `app/api/v1/routes/applications.py` — `GET /api/v1/applications/runs/{run_id}` added
- `alembic/` — migration setup; one migration creates the `application_tailoring_runs` table
- `tests/conftest.py` — autouse fixtures create a fresh in-memory SQLite DB per test and override `get_db`; all 60 tests run fully offline

**New endpoint:**
```
GET /api/v1/applications/runs/{run_id}
```
Returns `ApplicationTailoringRunResponse` — includes AI output, `provider_used`, `fallback_used`, and `created_at`.

**Environment variable:**
```
DATABASE_URL=sqlite:///./local.db          # default (local dev)
DATABASE_URL=postgresql+psycopg://...      # PostgreSQL (production)
```

**Migration commands:**
```bash
# Apply migrations (creates tables)
alembic upgrade head

# Check current migration state
alembic current

# Generate a new migration after model changes
alembic revision --autogenerate -m "describe your change"

# Roll back one migration
alembic downgrade -1
```

**Inspect stored runs (SQLite):**
```bash
sqlite3 local.db "SELECT id, provider_used, fallback_used, created_at FROM application_tailoring_runs;"
```

**What is intentionally not included:**
- User accounts / authentication — runs are not scoped to users yet
- pgvector / embeddings — a future milestone
- Pagination on the runs list endpoint — not added yet
- Async SQLAlchemy — synchronous is simpler and sufficient for this stage

---

---

## Milestone 6 — Background Job Processing

Converts the synchronous tailoring endpoint into an async workflow. The API
accepts requests immediately, enqueues generation as a background task, and
lets the caller poll for results — decoupling request latency from AI execution time.

**Why async workflows matter for AI systems:**
LLM generation is inherently slow (seconds to tens of seconds). A synchronous
API that blocks until generation finishes creates poor UX, exhausted thread
pools, and brittle timeouts. The industry pattern — used by OpenAI, Anthropic,
and every production AI pipeline — is to accept requests instantly, process
asynchronously, and expose status + results via a polling or webhook API.

**What's included:**
- `app/models/run_status.py` — `RunStatus` StrEnum: `pending`, `processing`, `completed`, `failed`
- `app/models/application.py` — two new columns (`status`, `error_message`); all AI output
  columns made nullable (they start as NULL for pending rows)
- `app/repositories/application_runs.py` — three new helpers:
  `create_pending_run`, `update_run_status`, `save_completed_run`
- `app/services/background_tailoring.py` — `process_tailoring_job(run_id, db)`:
  the background task that drives the full lifecycle
- Updated `POST /api/v1/applications/tailor` — creates the DB row, enqueues the task,
  returns `{run_id, status: "pending"}` immediately
- Updated `GET /api/v1/applications/runs/{run_id}` — returns output fields when
  completed, error_message when failed, null output when pending/processing
- Alembic migration `b2c3d4e5f6a7` — adds status/error_message columns, makes output
  columns nullable via batch mode (SQLite + PostgreSQL compatible)
- 14 new tests in `tests/test_background_jobs.py`

**Workflow lifecycle:**

```
POST /tailor
  └─ create row  (status=pending)
  └─ enqueue BackgroundTask
  └─ return {run_id, status="pending"}   ← instant response

BackgroundTask
  ├─ set status=processing
  ├─ build prompt, call LLM, parse JSON, apply fallback if needed
  ├─ success → persist output, set status=completed
  └─ failure → set status=failed, store error_message

GET /runs/{run_id}
  ├─ pending / processing → {id, status, created_at, output fields: null}
  ├─ completed            → {id, status, all output fields populated, ...}
  └─ failed               → {id, status, error_message, output fields: null}
```

**Background processing architecture:**

This milestone uses **FastAPI's built-in `BackgroundTasks`** — no external
broker, no Redis, no Celery. The task runs in the same process after the HTTP
response is sent. This is appropriate for a portfolio project and correctly
teaches the lifecycle pattern; a production system at scale would replace the
in-process task with a distributed worker queue (see "Not Included Yet").

**Session passing:** The background task receives the SQLAlchemy `Session`
from the route handler. In Starlette's request lifecycle, background tasks
execute after the response body is sent but before dependency teardown, so
the session remains valid throughout the task.

**Test behaviour:** Starlette's `TestClient` runs background tasks synchronously
before returning the response to the caller. Tests can therefore check the final
DB state immediately after `client.post(...)` returns, while the POST response
body still correctly reflects the initial `pending` status.

**New response schemas:**

```
POST /api/v1/applications/tailor
→ ApplicationTailoringJobResponse { run_id: int, status: str }

GET /api/v1/applications/runs/{run_id}
→ ApplicationTailoringRunResponse {
    id, status, error_message?,
    tailored_summary?, tailored_bullets?, ...,   ← null unless completed
    provider_used?, fallback_used, created_at
  }
```

**Migration commands:**

```bash
# Apply M6 migration (adds status/error_message, relaxes nullable constraints)
alembic upgrade head

# Verify
alembic current

# Roll back M6 only
alembic downgrade -1
```

**What is intentionally not included:**
- Redis / Celery / distributed workers — in-process BackgroundTasks is
  the right teaching tool at this stage
- Webhook callbacks — polling is simpler and sufficient
- WebSockets / server-sent events for live status updates
- Retry logic on failed jobs
- A `GET /runs` list endpoint with pagination

---

---

## Milestone 7 — Workflow Metadata Tracking

Adds lightweight instrumentation to every background generation job: timing,
token estimates, cost estimates, and attempt counts. No external observability
platform is needed — the metadata lives in the existing DB table and is exposed
via the existing GET endpoint.

**Why observability matters for AI systems:**
LLM calls are slow, expensive, and occasionally unreliable. Without
instrumentation you have no visibility into how long generation takes, whether
the fallback path is being exercised, or what a job actually cost. Even a
simple latency field transforms debugging from "it felt slow" to "that run
took 4 200 ms vs the median 800 ms — here's why."

**What's included:**
- `app/llm/token_estimation.py` — `estimate_input_tokens`, `estimate_output_tokens`
  using word-count approximation (`len(text.split())`). No tokenizer library.
- `app/llm/cost_estimation.py` — `estimate_generation_cost(input_tokens, output_tokens, provider)`
  with per-provider per-1K pricing constants. Mock = $0. Gemini ≈ industry pricing.
- `app/models/application.py` — 7 new nullable/defaulted columns on `ApplicationTailoringRun`
- `app/repositories/application_runs.py` — `save_completed_run` extended;
  `update_run_status` accepts optional timing params (used on failure)
- `app/services/background_tailoring.py` — records `started_at`, `completed_at`,
  `latency_ms`, token estimates, cost, and `generation_attempts` on every run
- `app/schemas/application.py` — all new fields exposed in `ApplicationTailoringRunResponse`
- Alembic migration `c3d4e5f6a7b8` — 7 `ADD COLUMN` statements (no batch mode needed)
- 36 new tests in `tests/test_workflow_metadata.py`

**New columns on `application_tailoring_runs`:**

| Column | Type | Description |
|---|---|---|
| `started_at` | DateTime nullable | When background task began executing |
| `completed_at` | DateTime nullable | When background task finished |
| `latency_ms` | Integer nullable | `(completed_at − started_at)` in ms |
| `estimated_input_tokens` | Integer nullable | Word-count approx of prompt tokens |
| `estimated_output_tokens` | Integer nullable | Word-count approx of output tokens |
| `estimated_cost_usd` | Float nullable | Rough USD cost based on provider pricing |
| `generation_attempts` | Integer default=0 | 1 = success; 2 = primary failed + fallback |

**generation_attempts logic:**

```
Primary provider succeeds   → attempts = 1
Primary fails, fallback runs → attempts = 2
Non-LLM exception before result → attempts = 1 (call was attempted)
Exception before LLM call       → attempts = 0
```

**Token estimation approach:**

```python
# app/llm/token_estimation.py
def estimate_input_tokens(prompt: str) -> int:
    return len(prompt.split())      # words ≈ tokens (rough)

def estimate_output_tokens(output: str) -> int:
    return len(output.split())
```

Real tokenizers (tiktoken, SentencePiece) are provider-specific and heavy.
A word-count approximation gives a useful order-of-magnitude signal without
the dependency. All cost figures are clearly marked as educational estimates.

**Cost estimation approach:**

```python
# Approximate USD per 1 000 tokens (Gemini Flash class, as of writing)
input:  $0.000075 / 1K tokens  (~$0.075 / 1M)
output: $0.000300 / 1K tokens  (~$0.30  / 1M)
mock / fallback-mock: $0.00
```

**Failed runs still capture partial metadata:**
`started_at`, `completed_at`, `latency_ms`, and `generation_attempts` are
written to the DB even when the job fails. This means debugging tools can
answer "how long did it run before failing?" without needing full output data.

**What is intentionally not included:**
- Real tokenizer libraries (tiktoken, SentencePiece) — word-count is sufficient
  for educational tracking; exact tokenization is provider-specific
- External observability platforms (Datadog, OpenTelemetry, Prometheus) — the
  data lives in the DB; a dashboard can be built from it later
- Structured logging / trace IDs — future milestone
- Percentile analytics / aggregation queries — future milestone

---

---

## Milestone 8 — Agentic Application Workflow

Introduces an optional **four-stage agentic workflow** powered by LangGraph.
Instead of a single LLM call that produces all output at once, the workflow
chains four focused calls — each reasoning about a narrower problem and passing
its output to the next stage. The single-step mode remains the default; agentic
mode is opt-in via a single environment variable.

**Why multi-stage reasoning?**
A single prompt asking the model to simultaneously analyse a resume, understand
a job description, identify gaps, and write application materials is asking it
to do too much at once. Breaking the task into sequential stages with
intermediate schema-validated outputs improves reliability and makes each
reasoning step auditable.

**What's included:**
- `app/schemas/agent.py` — three intermediate Pydantic schemas: `ResumeAnalysis`,
  `JobDescriptionAnalysis`, `FitGapAnalysis`
- `app/prompts/agentic_tailoring.py` — four prompt builders, each embedding a
  `## Task: <stage>` header so the mock provider returns the correct shape
- `app/services/agentic_tailoring.py` — LangGraph `StateGraph` with four nodes;
  each node makes one LLM call and parses the result into its schema; provider
  fallback is applied per-stage
- `app/llm/mock.py` — extended with stage-header detection and per-stage
  deterministic JSON responses; existing single-step behaviour unchanged
- `app/core/config.py` — new `workflow_mode` setting (`single_step` | `agentic`)
- `app/services/background_tailoring.py` — mode selection: `single_step` calls
  `_get_llm_output`; `agentic` calls `run_agentic_workflow`; unsupported modes
  store a `failed` run with a clear error message
- `.gitignore` — added `local.db`, `*.db`, `*.sqlite`, `*.sqlite3` (SQLite files
  were accidentally trackable before this milestone)
- 19 new tests in `tests/test_agentic_workflow.py`

**The four stages:**

```
Stage 1 — analyze_resume
  Input:  master_resume text
  Output: ResumeAnalysis { key_skills, relevant_experience, strengths }

Stage 2 — analyze_jd
  Input:  job_description text
  Output: JobDescriptionAnalysis { required_skills, responsibilities, role_focus }

Stage 3 — analyze_fit_gap
  Input:  ResumeAnalysis + JobDescriptionAnalysis
  Output: FitGapAnalysis { fit_points, gap_points, positioning_strategy }

Stage 4 — compose_final
  Input:  all three analyses + original job description
  Output: TailoringLLMOutput (same schema as single-step mode)
```

**Enabling agentic mode:**

```bash
# .env
WORKFLOW_MODE=agentic
```

Or keep the default:

```
WORKFLOW_MODE=single_step   ← default, preserves pre-M8 behaviour
```

**generation_attempts in agentic mode:**

```
4 stages, no fallback  → generation_attempts = 4
4 stages, all fell back → generation_attempts = 8
```

**Fallback behaviour:**
Each node independently applies provider fallback. If the configured LLM
provider fails for a stage, that stage (and all subsequent stages) fall back
to the mock provider. `fallback_used=True` is propagated to the run record.

**What is intentionally not included:**
- Parallel stage execution — the four stages are sequential because each stage
  consumes output from the previous one
- Conditional branching / retries at the graph level — straightforward chain
  is sufficient at this stage
- Streaming intermediate results — polling the final output is adequate here
- Redis / Celery / Docker — still in-process, no external dependencies added

---

## Not Included Yet (Intentionally)

- pgvector / embeddings
- Redis / Celery / distributed workers
- Authentication / user accounts
- Docker / CI/CD
- External observability platforms (Datadog, OpenTelemetry)
