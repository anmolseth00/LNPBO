.PHONY: install install-dev test lint format check figures experiments reproduce

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev,bench,mordred]"

test:
	uv run python -m pytest tests/ -x -q

lint:
	ruff check .

format:
	ruff format .

check: lint test

# ---------- Experiment reproduction ----------

# Main within-study benchmark (parallel, ~24h on 12-core machine)
experiments-within-study:
	./scripts/run_parallel.sh within-study

# Ablation experiments (run sequentially; each takes 2-12h)
experiments-ablations:
	./scripts/run_parallel.sh ablations

# Baseline comparisons
experiments-baselines:
	./scripts/run_parallel.sh baselines

# Sensitivity analyses
experiments-sensitivity:
	./scripts/run_parallel.sh sensitivity

# Cross-study transfer and calibration
experiments-analysis:
	./scripts/run_parallel.sh analysis

# All experiments (WARNING: takes 50+ hours of compute)
experiments: experiments-within-study experiments-ablations experiments-baselines experiments-sensitivity experiments-analysis

# ---------- Figure generation ----------

# Generate all publication figures (requires completed experiments)
figures:
	./scripts/run_parallel.sh figures

# ---------- Full reproduction ----------

# Run all experiments and regenerate all figures
reproduce: experiments figures
	cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
	cd paper && pdflatex -interaction=nonstopmode si.tex && pdflatex -interaction=nonstopmode si.tex
	@echo "Reproduction complete. PDFs at paper/main.pdf and paper/si.pdf"
