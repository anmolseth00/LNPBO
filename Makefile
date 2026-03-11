.PHONY: install install-dev test lint format check

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev,bench,mordred]"

test:
	python -m pytest tests/ -x -q

lint:
	ruff check .

format:
	ruff format .

check: lint test
