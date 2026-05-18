# Makefile — GA Strategy Evolution
# Usage: make <target>
#
# Common workflows:
#   make test            — run all GA tests
#   make smoke           — quick 2-min smoke test  
#   make validate-configs— check all configs for errors
#   make benchmark       — queue all 8 benchmark configs

SHELL := /bin/bash
PYTHON := python
GA := $(PYTHON) -m genetic_algorithm
PYTEST := $(PYTHON) -m pytest

# ─── Testing ────────────────────────────────────────────
.PHONY: test test-ga test-infra test-quick

test: ## Run all GA tests (1400+)
	$(PYTEST) genetic_algorithm/tests/ tests/test_ga_improvements.py \
		tests/test_engine_decomposition.py tests/test_runner.py \
		tests/test_generation_step.py tests/test_config_schema.py \
		tests/test_cli.py tests/test_registry.py \
		tests/test_orchestration.py tests/test_island_coordinator.py \
		-q --tb=short \
		--ignore=genetic_algorithm/tests/test_phase1a_llm_seeding.py \
		--ignore=genetic_algorithm/tests/test_overfitting_and_llm.py

test-ga: ## Run only genetic_algorithm/tests/
	$(PYTEST) genetic_algorithm/tests/ -q --tb=short \
		--ignore=genetic_algorithm/tests/test_phase1a_llm_seeding.py \
		--ignore=genetic_algorithm/tests/test_overfitting_and_llm.py

test-infra: ## Run infrastructure tests (engine, orchestration)
	$(PYTEST) tests/test_engine_decomposition.py tests/test_runner.py \
		tests/test_generation_step.py tests/test_config_schema.py \
		tests/test_cli.py tests/test_registry.py \
		tests/test_orchestration.py tests/test_island_coordinator.py -q --tb=short

test-quick: ## Run fast subset (~1s)
	$(PYTEST) tests/test_config_schema.py tests/test_registry.py tests/test_cli.py -q

# ─── GA Runs ────────────────────────────────────────────
.PHONY: smoke run benchmark

smoke: ## Quick smoke test (2 gen × 5 pop, ~2 min)
	$(GA) run quick_test --no-monitor --yes

run: ## Run with a config: make run CONFIG=path/to/config.yaml
	$(GA) run $(CONFIG) --no-monitor --yes

benchmark: ## Queue all 8 benchmark configs
	$(GA) queue add --dir genetic_algorithm/config/benchmark/ --tag benchmark

# ─── Config Management ──────────────────────────────────
.PHONY: validate-configs configs

validate-configs: ## Validate all preset and benchmark configs
	@echo "=== Presets ===" && \
	for f in genetic_algorithm/config/presets/*.yaml; do \
		$(GA) config validate "$$f" 2>/dev/null || true; \
	done && \
	echo "" && echo "=== Benchmarks ===" && \
	for f in genetic_algorithm/config/benchmark/*.yaml; do \
		$(GA) config validate "$$f" 2>/dev/null || true; \
	done

configs: ## List all available configs
	$(GA) config list --benchmarks

# ─── Queue & Monitor ────────────────────────────────────
.PHONY: queue-status monitor experiments

queue-status: ## Show queue status
	$(GA) queue status

monitor: ## Live monitor of running experiments
	$(GA) monitor --filter all --once

experiments: ## List recent experiments
	$(GA) experiment list --limit 20

# ─── Data Management ────────────────────────────────────
.PHONY: disk-report cleanup

disk-report: ## Show disk usage report
	$(GA) data report

cleanup: ## Clean up old data (dry-run)
	$(GA) data cleanup --dry-run

# ─── Development ────────────────────────────────────────
.PHONY: lint help

lint: ## Run basic Python checks
	$(PYTHON) -m py_compile genetic_algorithm/cli.py
	$(PYTHON) -m py_compile genetic_algorithm/core/evolution.py
	$(PYTHON) -m py_compile genetic_algorithm/engine/runner.py
	@echo "✓ Core modules compile OK"

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
