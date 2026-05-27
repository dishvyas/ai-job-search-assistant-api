# Workflow Evaluation Harness

This directory contains a lightweight, local-first evaluation harness for the AI
workflow outputs in this project. It is intentionally offline, deterministic, and
 interview-explainable.

## What it contains

- `cases/` — compact golden-style evaluation inputs
- `scoring.py` — deterministic output scoring helpers
- `run_eval.py` — CLI runner that executes evals against the FastAPI app with `TestClient`

## Eval case format

Each JSON case includes:

- `name`
- `description`
- `master_resume`
- `job_description`
- `company_info`
- `user_preferences`
- `expected_keywords`
- `must_not_include`
- `expected_sections`
- `notes`

The resume and job description content is fictional and compact on purpose so
cases stay safe, readable, and fast to run locally.

## How scoring works

`score_output()` performs deterministic checks only. There is no LLM judge.

Checks currently cover:

- required sections present
- `tailored_bullets` non-empty
- `interview_talking_points` non-empty
- `cover_letter_draft` non-empty
- `fit_gap_analysis` non-empty
- keyword coverage from `expected_keywords`
- forbidden text from `must_not_include`
- simple length sanity checks
- workflow metadata presence when metadata is provided

The scorer returns:

- `total_score`
- `max_score`
- `passed`
- `checks`

Each check includes:

- `name`
- `passed`
- `points`
- `max_points`
- `details`

## How to run

```bash
python evals/run_eval.py --workflow-mode single_step
python evals/run_eval.py --workflow-mode agentic
python evals/run_eval.py --workflow-mode single_step --case backend_engineer_germany
python evals/run_eval.py --compare
```

## Adding a new case

1. Add a new JSON file to `evals/cases/`
2. Keep the input fictional and compact
3. Choose `expected_keywords` that should reliably appear in acceptable output
4. Add anything sensitive or obviously wrong to `must_not_include`
5. Run the harness in both workflow modes to verify the case behaves as expected

## Limitations

- Deterministic scoring checks structure and basic content, not semantic quality
- Keyword coverage is shallow and can miss nuance
- Scores are local regression signals, not production truth
- The harness runs in mock mode, so it is best for workflow regression and plumbing checks

## Future improvements

- LLM-as-judge
- human review labels
- semantic similarity checks
- provider comparison beyond mock mode
- regression history over time
