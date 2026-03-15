# Autotrack Makefile
# Usage: make <target>

PYTHON  := python3
VENV    := .venv
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
PYTHON_VENV := $(VENV)/bin/python

.PHONY: help venv install install-dev test test-verbose export-coreml clean

help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv:          ## Create the virtual environment
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv  ## Install runtime dependencies
	$(PIP) install -e .

install-dev: venv  ## Install all dependencies including dev/test tools
	$(PIP) install -e ".[dev]"

test:  ## Run the full test suite
	$(PYTEST) tests/ -v

test-verbose:  ## Run tests with extra output
	$(PYTEST) tests/ -v --tb=long

export-coreml: install  ## Export yolov8n.pt to CoreML for Apple Neural Engine (2-4x faster than MPS)
	$(VENV)/bin/autotrack-export

clean:         ## Remove build artifacts and the virtual environment
	rm -rf $(VENV) build dist src/autotrack.egg-info __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
