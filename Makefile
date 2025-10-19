.PHONY: help install install-dev test lint format type-check clean build upload docs

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest tests/ --cov=scripts --cov-report=term-missing --cov-report=html

lint:  ## Run linting
	ruff check .

format:  ## Format code
	ruff format .

type-check:  ## Run type checking
	mypy scripts/

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ htmlcov/ .coverage .pytest_cache/ .ruff_cache/

build:  ## Build package
	python -m build

upload:  ## Upload to PyPI
	python -m twine upload dist/*

docs:  ## Build documentation
	mkdocs build

docs-serve:  ## Serve documentation locally
	mkdocs serve
