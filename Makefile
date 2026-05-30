test:
	./.venv/bin/pytest -q

lint:
	./.venv/bin/ruff check app tests evals alembic scripts

format:
	./.venv/bin/ruff format app tests evals alembic scripts

compose-up:
	docker compose up --build

compose-down:
	docker compose down

compose-migrate:
	docker compose exec api alembic upgrade head

compose-logs:
	docker compose logs -f api postgres

eval-mock-single:
	./.venv/bin/python evals/run_eval.py --provider mock --workflow-mode single_step

eval-mock-agentic:
	./.venv/bin/python evals/run_eval.py --provider mock --workflow-mode agentic
