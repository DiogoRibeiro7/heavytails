.DEFAULT_GOAL := help

POETRY ?= poetry
RUN    := $(POETRY) run

.PHONY: help install install-dev hooks test test-fast coverage lint format \
        format-check type-check security audit check docs docs-serve \
        build clean release-check release-preflight verify-release

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package only
	$(POETRY) install --only main

install-dev:  ## Install the package with all development dependency groups
	$(POETRY) install --with dev,docs,examples,benchmarks

hooks:  ## Install the pre-commit git hooks
	$(RUN) pre-commit install

test:  ## Run the test suite with coverage
	$(RUN) pytest tests/ --cov=heavytails --cov-report=term-missing --cov-report=html

test-fast:  ## Run the test suite in parallel, skipping slow tests
	$(RUN) pytest tests/ -m "not slow" -n auto --no-cov

coverage:  ## Run the test suite and enforce the minimum coverage threshold
	$(RUN) pytest tests/ --cov=heavytails --cov-report=term-missing --cov-fail-under=80

lint:  ## Run the linter
	$(RUN) ruff check .

format:  ## Format the code in place
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

format-check:  ## Verify formatting without modifying files
	$(RUN) ruff format --check .

type-check:  ## Run static type checking
	$(RUN) mypy heavytails/ scripts/

security:  ## Run the static security linter
	$(RUN) bandit -c pyproject.toml -r heavytails/ scripts/

audit:  ## Check installed dependencies for known vulnerabilities
	$(RUN) pip-audit --skip-editable

check: lint format-check type-check test security release-check  ## Run every check that CI runs

release-check:  ## Verify the five files agree on which release this is
	$(RUN) python -m scripts.check_release

release-preflight:  ## release-check, plus refuse a tag that is already spent
	$(RUN) python -m scripts.check_release --pre-tag

verify-release:  ## Verify VERSION=x.y.z reached GitHub, PyPI and Zenodo
	$(RUN) python -m scripts.verify_release $(VERSION)

docs:  ## Build the documentation (fails on broken links)
	$(RUN) mkdocs build --strict

docs-serve:  ## Serve the documentation with live reload
	$(RUN) mkdocs serve

build:  ## Build the sdist and wheel
	$(POETRY) build

clean:  ## Remove build, test and cache artifacts
	rm -rf build/ dist/ site/ htmlcov/ .coverage coverage.xml coverage.json \
	       .pytest_cache/ .ruff_cache/ .mypy_cache/ .hypothesis/ .benchmarks/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
