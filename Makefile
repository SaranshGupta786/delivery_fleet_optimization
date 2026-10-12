# Delivery-fleet allocation engine - docker-first workflow.
#
#   make up        build the image and start API+UI on http://localhost:8000
#   make test      run the pytest suite inside the image (same deps as runtime)
#   make smoke     hit the running container end to end (health, allocate, compare)
#   make bench     re-run the strategy benchmark and rewrite ANALYSIS.md
#   make train     retrain the RL policy artifact inside the image
#
# Local (no Docker) equivalents are provided for every target with a -local suffix.

SHELL := /bin/bash
IMAGE ?= delivery-fleet-allocation-engine
TAG ?= 1.0.0
REF := $(IMAGE):$(TAG)
TEST_REF := $(IMAGE):$(TAG)-test
COMPOSE ?= docker compose
PORT ?= 8000
PYTHON ?= python3
VENV ?= .venv
VENV_PY := $(VENV)/bin/python

.PHONY: help build up down restart logs ps test test-local lint frontend train train-local bench bench-local capacity capacity-local smoke smoke-local clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build the production image
	docker build --tag $(REF) --target runtime .

up: build ## Build and start the container (API + UI on one port)
	$(COMPOSE) up --detach --build
	@echo "waiting for http://localhost:$(PORT)/api/health ..."
	@for i in $$(seq 1 40); do \
		if curl -fsS http://localhost:$(PORT)/api/health >/dev/null 2>&1; then \
			echo "ready: http://localhost:$(PORT)"; exit 0; \
		fi; \
		sleep 1; \
	done; \
	echo "service did not become healthy in time"; $(COMPOSE) logs --tail 40; exit 1

down: ## Stop the container
	$(COMPOSE) down --remove-orphans

restart: down up ## Recreate the container

logs: ## Follow container logs
	$(COMPOSE) logs --follow

ps: ## Show container status
	$(COMPOSE) ps

test: ## Run the pytest suite inside the image (builds the test stage)
	docker build --tag $(TEST_REF) --target test .
	docker run --rm $(TEST_REF)

test-local: ## Run the pytest suite against the local virtualenv
	cd backend && PYTHONPATH=. ../$(VENV_PY) -m pytest -q

frontend: ## Build the SPA into frontend/dist (needs Node 20)
	cd frontend && npm ci --no-audit --no-fund && npm run build

train: build ## Retrain the RL policy artifact inside the image
	docker run --rm --user root -v "$$(pwd)/backend/artifacts:/app/backend/artifacts" $(REF) \
		python -m app.rl.train --episodes 2500 --out /app/backend/artifacts/rl_policy.json

train-local: ## Retrain the RL policy artifact locally
	cd backend && PYTHONPATH=. ../$(VENV_PY) -m app.rl.train --episodes 2500 \
		--out artifacts/rl_policy.json

bench: build ## Run the benchmark and rewrite ANALYSIS.md
	docker run --rm --user root -v "$$(pwd):/work" -w /app/backend -e BENCH_OUT=/work/ANALYSIS.md \
		$(REF) python -m app.bench

bench-local: ## Run the benchmark locally
	cd backend && PYTHONPATH=. BENCH_OUT=../ANALYSIS.md ../$(VENV_PY) -m app.bench

capacity: build ## Sweep capacity pressure and write CAPACITY_SWEEP.md
	docker run --rm --user root -v "$$(pwd):/work" -w /app/backend -e BENCH_OUT=/work/CAPACITY_SWEEP.md \
		$(REF) python -m app.capacity_sweep

capacity-local: ## Sweep capacity pressure locally
	cd backend && PYTHONPATH=. BENCH_OUT=../CAPACITY_SWEEP.md ../$(VENV_PY) -m app.capacity_sweep

smoke: ## End-to-end smoke test against a running server
	BASE_URL=$${BASE_URL:-http://localhost:$(PORT)} ./scripts/smoke.sh

clean: ## Remove the built images and local caches
	-$(COMPOSE) down --remove-orphans
	-docker image rm $(REF) $(TEST_REF)
	rm -rf frontend/dist backend/.pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
