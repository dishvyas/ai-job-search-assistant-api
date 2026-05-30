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

Supported generation providers now include:
- `mock`
- `gemini`
- `openai`

**New environment variables:**

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock`, `gemini`, or `openai` |
| `GEMINI_API_KEY` | _(empty)_ | Required only when `LLM_PROVIDER=gemini` |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model name |
| `OPENAI_API_KEY` | _(empty)_ | Required when `LLM_PROVIDER=openai`; also reused for embeddings |
| `OPENAI_MODEL` | `gpt-4.1-mini` | OpenAI generation model |

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

**Example `.env` for OpenAI generation:**

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4.1-mini
RAG_ENABLED=false
```

The same `OPENAI_API_KEY` is also used for embeddings when `RAG_ENABLED=true`.
RAG still requires PostgreSQL with `pgvector` for meaningful vector similarity search.

## Real-provider local testing

You can verify real OpenAI generation first without involving RAG:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
RAG_ENABLED=false
```

That setup works for both single-step and agentic generation without any vector
database dependency.

For an embedding smoke test:

```bash
python - <<'PY'
from app.rag.embed import generate_embedding
vec = generate_embedding("AI backend engineer FastAPI RAG evaluation")
print(len(vec))
PY
```

RAG ingest requires all of the following:

- PostgreSQL
- `pgvector`
- `OPENAI_API_KEY` with embeddings quota/billing
- `RAG_ENABLED=true`

If `POST /api/v1/jobs/ingest` returns `503 Service Unavailable`, check:

- API quota/billing
- `OPENAI_API_KEY`
- `EMBEDDING_MODEL`
- model access for the configured embedding model

## Milestone — Agent Stage Cost Metadata

Workflow-level metadata is useful for seeing the total latency, approximate
tokens, and rough cost of a completed run. Once the workflow becomes agentic,
that total is not enough to explain where the expensive work happened.

This milestone adds approximate per-stage metadata to
`GET /api/v1/applications/runs/{run_id}/trace`:

- `estimated_input_tokens`
- `estimated_output_tokens`
- `estimated_cost_usd`

Why this matters:

- workflow-level metadata answers "what did this run cost overall?"
- agent-stage metadata answers "which node spent the tokens and time?"
- stage-level estimates make it easier to debug expensive prompts, large
  revision passes, and multi-call agent behavior

These values are intentionally approximate. Exact billing would require
provider-native usage metadata or provider-specific tokenizer libraries, which
this project intentionally avoids to keep the implementation lightweight and
interview-explainable.

Raw prompts and raw model responses are still not stored in traces. The trace
table continues to store only compact summaries plus observability metadata.

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

---

## Milestone 9 — RAG Job Matching

Introduces an optional **Retrieval-Augmented Generation (RAG) pipeline** for job description matching. When enabled, the most semantically similar stored job descriptions are retrieved and injected into the tailoring prompt — providing real-world vocabulary, skill requirements, and role context that improves the specificity of generated application materials. RAG is **disabled by default**; all existing workflow behaviour is fully preserved.

**Why RAG improves tailoring quality:**
Prompting an LLM with only a single job description is limited — the model reasons from its training data, which may not reflect current market language or the specific skills your target company values. Retrieved similar roles provide concrete grounding: the prompt can say "here are three roles where this skill matters and is phrased this way" rather than asking the model to guess from scratch.

**What's included:**
- `app/rag/embed.py` — `generate_embedding(text)` wraps the OpenAI embeddings API
- `app/rag/ingest.py` — `ingest_job_description(db, ...)` embeds `raw_text` and stores the record
- `app/rag/retrieve.py` — `retrieve_relevant_jobs(db, query, ...)` with query enrichment, cosine similarity search, metadata filters, and similarity threshold
- `app/rag/eval.py` — `score_retrieval(query, retrieved_jobs)` — keyword-overlap coverage signal
- `app/models/job_description.py` — `JobDescription` ORM model with `Vector(1536)` embedding column
- `app/schemas/jobs.py` — request/response schemas for jobs routes
- `app/api/v1/routes/jobs.py` — three new routes: `/ingest`, `/match`, `/compare`
- `app/prompts/tailoring.py` — updated to accept optional `rag_context` parameter
- `app/llm/mock.py` — RAG-enriched prompt detection + deterministic `[MOCK-RAG]` response
- Alembic migration `d4e5f6a7b8c9` — creates `job_descriptions` table with pgvector extension
- 21 new tests in `tests/test_rag.py`

**Enabling RAG locally (requires PostgreSQL + pgvector):**

```bash
# .env
RAG_ENABLED=true
OPENAI_API_KEY=your-key-here
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/dbname

# Apply migration (creates pgvector extension + job_descriptions table)
alembic upgrade head
```

**Ingestion flow:**

```
POST /api/v1/jobs/ingest
  body: { title, company, location, raw_text, metadata }
  → generate_embedding(raw_text)          # OpenAI API call
  → INSERT INTO job_descriptions          # stores text + vector
  → return { job_description_id, title, created_at }
```

We embed `raw_text` rather than just the `title` because the title alone ("Senior Engineer") is too generic. Skills, responsibilities, and requirements in the body text carry the semantic signal needed for meaningful similarity.

**Retrieval flow:**

```
POST /api/v1/jobs/match
  body: { query, filters (optional), top_k (optional) }
  → enrich query with filter context         # improves vector specificity
  → generate_embedding(enriched_query)
  → SELECT ... ORDER BY embedding <=> query  # cosine similarity via pgvector
  → apply metadata filters (WHERE)           # precision improvement
  → discard matches below similarity_threshold
  → return ranked list with similarity scores
```

**Before/after compare endpoint:**

```
POST /api/v1/jobs/compare
  body: { query, resume_summary }
  → LLM call 1: plain prompt (no RAG)       # without_rag response
  → retrieve_relevant_jobs(query)            # if RAG enabled
  → LLM call 2: prompt + retrieved context  # with_rag response
  → return { without_rag, with_rag, retrieved_jobs }
```

**Key design decisions:**

| Decision | Why |
|---|---|
| pgvector over dedicated vector DB | Keeps the stack simple — one database for relational data and vectors. For this scale, pgvector's ANN search is fast enough. A dedicated vector store (Pinecone, Weaviate) adds operational overhead without meaningful quality gain at this scale. |
| Query enrichment before embedding | A plain query ("Python backend engineer") embeds to a generic vector. Appending filter context ("role_type: backend, seniority: senior") shifts the vector toward the specific semantic neighbourhood, reducing false matches. |
| Semantic + metadata filters combined | Semantic search alone can return plausible-but-wrong results (e.g. a data engineering role when looking for backend). Metadata filters act as hard constraints that eliminate whole categories before distance ranking — improves precision without hurting recall on the filtered set. |
| Similarity threshold | Returning the top-k unconditionally is dangerous when the corpus is small or the query is unusual. A low-similarity "best match" that is still semantically distant from the query adds noise to the prompt, which can actively mislead the LLM. |
| Embed raw_text not just title | A title like "Senior Engineer" has ~0 discriminative power. The full JD text contains the skills, stack, responsibilities, and culture signals that determine whether a role is actually similar. |
| RAG disabled by default | The existing tailoring workflow is unaffected when `RAG_ENABLED=false`. No embedding API calls happen, no pgvector extension is required, SQLite still works for local dev and tests. |

**Known limitations and what production RAG would add:**
- No incremental index update — currently re-embeds on ingest; production would batch-update
- No ANN index (IVFFlat/HNSW) — full scan works for small corpora, becomes expensive at scale
- Token budget management — the current implementation truncates retrieved text at 500 chars; production would use a token budget tied to the model's context window
- Eval is keyword-overlap only — production would use human-labelled pairs + nDCG/MRR
- No deduplication on ingest — the same JD can be stored multiple times
- OpenAI embeddings only — a production system might cache embeddings or use a self-hosted model

**What is intentionally not included:**
- Async embedding generation (ingestion is synchronous)
- Vector index creation in migration (left for production tuning)
- Embedding caching (acceptable simplification at this stage)

---

## Milestone 10 — Tool-Using Agent Workflow with Review/Revision

Upgrades the agentic workflow so it behaves more like a real AI agent system: it **retrieves external context as a tool**, applies **routing logic** based on that context, produces output, then runs a lightweight **review/revision cycle** before returning a result. The workflow runs whenever `WORKFLOW_MODE=agentic` — no new configuration knobs needed.

**Why this goes beyond simple prompt chaining:**
The previous agentic workflow (M8) was sequential prompt chaining with a fixed four-stage pipeline. M10 adds three properties that distinguish agent workflows from chains:

1. **Tool use** — the `retrieve_context` node queries an external store (the vector DB) before any reasoning begins, incorporating real-world data into the agent's working context.
2. **Routing decisions** — the `decide_route` node inspects intermediate state (fit/gap ratio, available context) and labels the run with a decision. This demonstrates how agents can branch based on what they observe, not just execute a fixed sequence.
3. **Critique/revision loop** — the `review_output` node checks the final output for completeness, and the `revise_output` node corrects it if needed. This is the "reflection" pattern: the agent evaluates its own output and improves it — without human intervention.

**Updated workflow (8 nodes):**

```
START
  → retrieve_context      [tool use: RAG retrieval]
  → analyze_resume        [LLM call]
  → analyze_jd            [LLM call]
  → analyze_fit_gap       [LLM call — enriched with retrieved context]
  → decide_route          [deterministic: sets route_decision label]
  → compose_final         [LLM call — enriched with retrieved context]
  → review_output         [deterministic: checks output completeness]
  → [if revision_needed]
      revise_output       [LLM call — corrects incomplete output]
  → END
```

**Context retrieval node (`retrieve_context`):**
When `RAG_ENABLED=true` and a DB session is available, the workflow queries the vector DB using the job description text as the query. Retrieved snippets (up to 3, truncated at 300 chars each) are stored in `retrieved_context` state and injected into the fit/gap and final composition prompts. If retrieval is disabled, unavailable, or fails — the workflow continues with an empty context list. No exceptions propagate from this node.

**How context is used in prompts:**
Retrieved snippets are clearly labeled as `## Retrieved Context (from similar roles — for reference only)`. The prompt explicitly instructs the LLM to treat them as reference material, not as override instructions. The candidate's resume and job description remain the primary sources of truth.

**Routing decision node (`decide_route`):**
Deterministic logic — no LLM call:

| Condition | Route |
|---|---|
| No retrieved context AND no `company_info` | `needs_more_context` |
| `gap_points > fit_points` (from fit/gap analysis) | `low_fit_warning` |
| Otherwise | `proceed_to_tailoring` |

Routes are **advisory metadata** — the workflow always proceeds to composition regardless of the route decision. Hard-gating (e.g., refusing to generate for low-fit candidates) is a product-level decision that doesn't belong in a generic tailoring service. The route label is available for callers that want to surface it in a UI.

**Review node (`review_output`):**
Checks three structural conditions:
- `tailored_summary` is non-empty
- `tailored_bullets` is non-empty
- `interview_talking_points` is non-empty

If all pass → `revision_needed = False`, `review_notes = "Review passed..."`.
If any fail → `revision_needed = True`, `review_notes` describes what was missing.

Review is **deterministic** (no LLM call): cheaper, faster, fully auditable. The structural check catches the most common failure mode — a truncated or malformed LLM output missing required sections.

**Revision node (`revise_output`):**
Called only when `revision_needed = True`. Makes a single LLM call with the current (incomplete) output and the review notes, producing a corrected `TailoringLLMOutput`. After revision, the graph always goes to `END` — no second review is performed. This is intentional: at most one revision pass avoids any possibility of a correction loop.

**Updated state fields:**

```python
class AgenticTailoringState(TypedDict):
    request: ApplicationTailorRequest
    db: Any                    # SQLAlchemy session | None — for RAG tool use
    resume_analysis: ...       # M8: unchanged
    jd_analysis: ...           # M8: unchanged
    fit_gap: ...               # M8: unchanged
    final_output: ...          # M8: unchanged
    retrieved_context: list[str]   # NEW: text snippets from RAG
    route_decision: str            # NEW: advisory routing label
    review_notes: str | None       # NEW: outcome of the quality review
    revision_needed: bool          # NEW: whether revision was triggered
    provider_used: str
    fallback_used: bool
```

**Key tradeoffs:**

| Aspect | Tradeoff |
|---|---|
| More LLM calls | Up to 5 calls (4 stages + 1 revision) vs 4 previously. Higher cost and latency per request. Worth it when output completeness matters more than speed. |
| More explainable and controllable | Every decision (route, review pass/fail, revision trigger) is logged in state and can be inspected. This is the opposite of a black-box call to a single large model. |
| Deterministic routing and review | No extra LLM spend for decisions that can be made with logic. Keeps the cost increase predictable: only composition and (rarely) revision use the LLM. |
| No real looping | The graph prevents revision loops by design. A production system might allow multiple revision rounds, but that requires convergence guarantees that are out of scope here. |
| Context is optional | When RAG is disabled or retrieval returns nothing, all seven nodes run exactly as before. No behavioral change for `RAG_ENABLED=false` deployments. |

**This is controlled orchestration, not a fully autonomous agent:**
The graph structure, transition logic, and decision rules are all hardcoded in Python. The LLM cannot add nodes, modify the workflow, or call tools it wasn't given. This is intentional — for a production portfolio project, predictability and debuggability matter more than autonomy. The architecture demonstrates the key agent patterns (tool use, routing, reflection) without the risks of an open-ended agent that can take arbitrary actions.

**What was intentionally not implemented:**
- Conditional routing that skips composition for `low_fit_warning` (product decision out of scope)
- Semantic review via LLM-as-judge (cost and latency tradeoff not justified here)
- Multiple revision rounds (would require convergence logic)
- Exposing `route_decision` and `review_notes` in the API response (schema change deferred)
- Parallel node execution (retrieval could run in parallel with analysis; not needed at this scale)
- 30 new tests in `tests/test_agent_tools_review.py`

---

## Not Included Yet (Intentionally)

- Redis / Celery / distributed workers
- Authentication / user accounts
- Docker / CI/CD
- External observability platforms (Datadog, OpenTelemetry)

---

## Milestone 11 — Agent Workflow Traceability

Adds lightweight per-stage tracing for the `agentic` workflow so a completed run can be inspected after the fact without exposing raw prompts or PII-heavy source text.

**Why traceability matters:**
- Makes agent behavior observable instead of black-box
- Helps debug intermediate reasoning stages and fallback paths
- Gives production-style per-stage metadata without rewriting the existing graph
- Improves explainability for UIs, QA, and workflow audits

**What was added:**
- `agent_trace_steps` table for persisted stage-level workflow traces
- `AgentTraceStep` SQLAlchemy model and repository helpers
- `GET /api/v1/applications/runs/{run_id}/trace`
- Best-effort tracing inside agentic nodes:
  - `retrieve_context`
  - `analyze_resume`
  - `analyze_jd`
  - `analyze_fit_gap`
  - `decide_route`
  - `compose_final`
  - `review_output`
  - `revise_output` when revision runs

**Trace data includes:**
- `step_name`
- `status`
- short `input_summary`
- short `output_summary`
- `provider_used`
- `fallback_used`
- `latency_ms`
- `error_message` when a step fails
- `created_at`

**Trace data intentionally does not include:**
- raw prompts
- full resume text
- full job description text
- full retrieved context payloads

This keeps the trace useful for debugging while avoiding direct exposure of sensitive or high-volume inputs.

**New endpoint:**

```bash
GET /api/v1/applications/runs/{run_id}/trace
```

**Response shape:**

```json
{
  "run_id": 123,
  "steps": [
    {
      "id": 1,
      "run_id": 123,
      "step_name": "retrieve_context",
      "status": "completed",
      "input_summary": "RAG enabled=false; db available=true.",
      "output_summary": "Retrieved 0 context snippets.",
      "provider_used": "mock",
      "fallback_used": false,
      "latency_ms": 0,
      "error_message": null,
      "created_at": "2026-05-27T10:00:00Z"
    }
  ]
}
```

**Important implementation choices:**
- Tracing is agentic-only in this milestone; `single_step` runs return an empty trace list
- Trace persistence is best-effort; if a trace insert fails, the workflow still completes
- The existing single-step workflow, RAG behavior, mock mode, fallback behavior, and LangGraph structure are preserved

---

## Milestone 12 — Workflow Evaluation Harness

Adds a lightweight local evaluation harness for comparing workflow quality and reliability across `single_step` and `agentic` modes without any external eval service.

**Why evals matter:**
- provide golden regression checks for AI workflow behavior
- catch broken output structure before it reaches runtime users
- compare quality, cost, latency, and attempt-count tradeoffs between workflows
- keep workflow changes interview-friendly and explainable with deterministic scoring

**How to run evals:**

```bash
python evals/run_eval.py --workflow-mode single_step
python evals/run_eval.py --workflow-mode agentic
python evals/run_eval.py --compare
```

You can also scope to a single case:

```bash
python evals/run_eval.py --workflow-mode single_step --case backend_engineer_germany
```

**What scoring checks exist:**
- required output sections present
- `tailored_bullets` non-empty
- `interview_talking_points` non-empty
- `cover_letter_draft` non-empty
- `fit_gap_analysis` non-empty
- keyword coverage from case expectations
- forbidden text detection from `must_not_include`
- simple length sanity checks
- workflow metadata presence when available

**What is intentionally not included yet:**
- LLM-as-judge
- semantic scoring
- external eval platforms
- real production benchmark datasets

---

## Milestone 13 — Tailored Artifact Retrieval

Extends the RAG pipeline so the app can optionally retrieve prior generated tailoring artifacts, not just raw job descriptions. This is disabled by default and remains best-effort.

**Why tailored-artifact retrieval matters:**
- raw job descriptions mostly provide role requirements and vocabulary
- past tailored outputs capture positioning patterns, tone, and output structure
- similar generated artifacts can be more useful than generic JDs when the goal is better application materials, not just requirement matching

**Job-description RAG vs tailored-artifact RAG:**
- Job-description RAG retrieves similar roles and requirement language from stored JDs
- Tailored-artifact RAG retrieves prior generated summaries, bullets, fit/gap framing, interview points, and recruiter messaging
- Job-description RAG is about role context
- Tailored-artifact RAG is about reusable positioning style and structure

**Guardrails for artifact retrieval:**
- use retrieved artifacts for tone, structure, and positioning inspiration only
- do not copy claims
- do not invent experience not present in the current resume
- the current master resume and job description remain the source of truth

**What gets indexed as an artifact:**
- `tailored_summary`
- `tailored_bullets`
- `fit_gap_analysis`
- `interview_talking_points`
- `recruiter_message_draft`

**What does not get indexed:**
- full `master_resume`
- full `job_description`
- raw prompts
- trace data

**Operational notes:**
- artifact retrieval is disabled by default
- artifact indexing is best-effort after a completed run is saved
- if artifact embedding generation fails, the run stays completed
- when `RAG_ENABLED=false`, existing behavior remains unchanged

---

## Milestone 14 — Real-Provider Eval Reports

Extends the local eval harness so it can optionally run against a real configured provider and save structured JSON reports locally for later review.

**What this adds:**
- `--provider mock|gemini` on `evals/run_eval.py`
- `--save-report` to persist JSON reports under `evals/reports/`
- optional eval-time flags for `--rag-enabled` and `--artifact-retrieval-enabled`
- local report files that capture score, latency, cost, attempts, provider, fallback, and workflow mode

**Example commands:**

```bash
python evals/run_eval.py --provider mock --workflow-mode single_step
python evals/run_eval.py --provider mock --workflow-mode agentic
GEMINI_API_KEY=... python evals/run_eval.py --provider gemini --workflow-mode single_step --save-report
GEMINI_API_KEY=... python evals/run_eval.py --provider gemini --workflow-mode agentic --save-report
GEMINI_API_KEY=... python evals/run_eval.py --provider gemini --compare --save-report
OPENAI_API_KEY=... python evals/run_eval.py --provider openai --workflow-mode single_step --save-report
OPENAI_API_KEY=... python evals/run_eval.py --provider openai --workflow-mode agentic --save-report
OPENAI_API_KEY=... python evals/run_eval.py --provider openai --compare --save-report
```

**Important notes:**
- real-provider evals are manual only
- real-provider evals may cost money
- the runner fails fast if `--provider gemini` is used without `GEMINI_API_KEY`
- automated tests remain mock-only and do not call real Gemini or OpenAI

---

## Milestone 15 — Agent Decision Metadata

Adds a compact, safe agent decision summary to `GET /api/v1/applications/runs/{run_id}` for agentic runs.

**What this exposes:**
- `route_decision`
- `review_notes`
- `revision_needed`
- `retrieved_context_count`
- `artifact_context_count`

**Why this exists:**
- the main run endpoint can now show a concise explainability summary for product/UI use
- the trace endpoint still exists for stage-by-stage debug detail
- compact decision metadata is easier to surface than full internal traces

---

## Milestone 17 — Fallback Reason Metadata

Adds a safe `fallback_reason` field to completed runs so successful-but-degraded
responses explain why mock fallback was used.

**What this adds:**
- `fallback_reason` on `GET /api/v1/applications/runs/{run_id}`
- `fallback_reason` in saved eval report results
- safe degraded-success observability for provider and parsing failures

**Why this matters:**
- `error_message` is for failed runs
- `fallback_reason` is for completed runs that succeeded via fallback
- this makes it much easier to diagnose provider quota, timeout, model access,
  or parsing problems without confusing degraded success with a true failure

**Example degraded-success metadata:**
- `provider_used="fallback-mock"`
- `fallback_used=true`
- `fallback_reason="LLMProviderUnavailableError: OpenAI request failed"`

The fallback reason is intentionally short and sanitized:
- no raw prompts
- no raw model outputs
- no stack traces
- no API keys or secrets

**Run summary vs trace endpoint:**
- `GET /runs/{run_id}` now returns a compact workflow summary
- `GET /runs/{run_id}/trace` still returns per-stage debug detail

**Why these fields are safe to expose:**
- they are summaries and counters, not raw retrieved snippets
- they do not expose prompts
- they do not expose full resumes or full job descriptions
- they do not expose raw artifact examples

**Single-step behavior:**
- single-step runs return `null` for these agent-specific fields
