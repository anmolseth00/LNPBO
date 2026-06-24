.PHONY: install install-dev lint format check

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev,bench,mordred]"

lint:
	ruff check .

format:
	ruff format .

check: lint
