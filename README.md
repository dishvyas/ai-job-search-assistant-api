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

## Not Included Yet (Intentionally)

- Database (PostgreSQL / pgvector)
- Redis
- AI/LLM integration (Gemini, OpenAI)
- LangGraph workflow orchestration
- Authentication
- Background jobs
- Docker / CI/CD
