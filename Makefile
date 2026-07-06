.PHONY: help install test lint format type-check pre-commit clean build server quickstart sync dev-setup ci check install-pre-commit docs docs-serve docs-versions

# Include local overrides if present (not tracked in git)
-include Makefile.local

sync:
	uv sync --all-extras --all-groups

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in the current environment
	pip install -e .

test: ## Run tests with coverage
	uv run pytest

lint: ## Run linting with ruff
	uv run ruff check . && uv run ruff format --check .

format: ## Format code with ruff
	uv run ruff format .
	uv run ruff check --fix .

type-check: ## Run type checking with mypy
	MYPYPATH=src uv run mypy src/tac getting_started

pre-commit: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

install-pre-commit: ## Install pre-commit hooks
	uv run pre-commit install

check: lint type-check test ## Run all checks (lint, type-check, test)

clean: ## Clean up cache and build artifacts
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	uv build

setup: ## Start the Twilio setup wizard on port 8080
	uv run --with fastapi --with uvicorn python getting_started/twilio_setup/server.py

docs: ## Build the API documentation into site/
	uv run --group docs mkdocs build

docs-serve: ## Serve the docs locally with live reload at http://127.0.0.1:8000
	uv run --group docs mkdocs serve

docs-versions: ## List the versioned docs published to the gh-pages branch
	uv run --group docs mike list

dev-setup: sync install-pre-commit ## Complete development environment setup

ci: check ## Run CI checks locally