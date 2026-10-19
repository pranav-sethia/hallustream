.PHONY: test lint format smoke split

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

format:
	uv run ruff check --fix .
	uv run ruff format .

smoke:
	uv run modal run modal_jobs/smoke.py

split:
	uv run python -m hallustream.research.data data/splits.json
