.PHONY: help install dev test lint build docker-up docker-down clean

PYTHON ?= python3
VENV   ?= .venv
PIP    ?= $(VENV)/bin/pip
PY     ?= $(VENV)/bin/python

help:
	@echo "GHOST-SIGINT Makefile"
	@echo ""
	@echo "  make install     — create venv and install all dependencies"
	@echo "  make dev         — run backend in dev mode (hot reload)"
	@echo "  make test        — run full test suite with coverage"
	@echo "  make lint        — run ruff + mypy"
	@echo "  make build       — build Docker images"
	@echo "  make docker-up   — start all services with docker-compose"
	@echo "  make docker-down — stop all services"
	@echo "  make clean       — remove venv, __pycache__, .pytest_cache"

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

dev:
	$(PY) -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest backend/tests/ -v --tb=short --cov=backend --cov-report=term-missing

lint:
	$(VENV)/bin/ruff check backend/
	$(VENV)/bin/mypy backend/ --ignore-missing-imports

build:
	docker build -t ghost-sigint-backend -f Dockerfile.backend .
	docker build -t ghost-sigint-nginx   -f Dockerfile.nginx .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	rm -rf $(VENV) .pytest_cache htmlcov .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
