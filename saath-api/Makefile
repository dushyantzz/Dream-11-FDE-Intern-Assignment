.PHONY: help install run test lint fmt check clean docker-build docker-run replay

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PYTHON     ?= python
UV         ?= uv
APP_MODULE  = app.main:app
HOST        ?= 0.0.0.0
PORT        ?= 8000
IMAGE_NAME  ?= saath-api
IMAGE_TAG   ?= latest

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
install: ## Install all dependencies (uv)
	$(UV) sync --all-extras

# ---------------------------------------------------------------------------
# Development server
# ---------------------------------------------------------------------------
run: ## Run dev server with hot-reload
	$(UV) run uvicorn $(APP_MODULE) \
		--host $(HOST) \
		--port $(PORT) \
		--reload \
		--log-level warning

# ---------------------------------------------------------------------------
# Linting (Rule 2, 5, 6 enforcement via ruff + mypy)
# ---------------------------------------------------------------------------
lint: ## Run ruff + mypy
	$(UV) run ruff check app/ tests/
	$(UV) run ruff format --check app/ tests/
	$(UV) run mypy app/ --ignore-missing-imports

fmt: ## Auto-fix ruff lint issues and format
	$(UV) run ruff check --fix app/ tests/
	$(UV) run ruff format app/ tests/

# ---------------------------------------------------------------------------
# Tests (Rule 7: only unit tests by default; no network)
# ---------------------------------------------------------------------------
test: ## Run unit tests (no integration, no network)
	$(UV) run pytest -m "not integration" $(ARGS)

test-cov: ## Run tests with HTML coverage report
	$(UV) run pytest -m "not integration" --cov-report=html $(ARGS)

test-integration: ## Run integration tests (requires real credentials in .env)
	$(UV) run pytest -m "integration" $(ARGS)

# ---------------------------------------------------------------------------
# Match Replay
# ---------------------------------------------------------------------------
replay: ## Replay match to stdout (e.g. make replay FILE=../odis_json/919615.json)
	$(PYTHON) -m app.adapters.cricsheet $(if $(FILE),$(FILE),../odis_json/919615.json) --speed $(if $(SPEED),$(SPEED),1000)

# ---------------------------------------------------------------------------
# Combined CI gate
# ---------------------------------------------------------------------------
check: lint test ## Run lint + test (CI entry point)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean: ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"       -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build: ## Build the Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run: ## Run the Docker image locally
	docker run --rm -p $(PORT):$(PORT) \
		--env-file .env \
		$(IMAGE_NAME):$(IMAGE_TAG)
