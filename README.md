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

## Not Included Yet (Intentionally)

- pgvector / embeddings
- Redis / Celery / distributed workers
- LangGraph workflow orchestration
- Authentication / user accounts
- Docker / CI/CD
