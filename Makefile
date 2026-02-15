.PHONY: help install install-dev lint format type-check test test-cov test-schema test-api test-stacking check train clean docker-build docker-run docker-test mlflow-ui validate-data oof-all stacking

PYTHON := python3
PROJECT_NAME := heart-disease-prediction

help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make lint         - Run linter (ruff)"
	@echo "  make format       - Format code (ruff)"
	@echo "  make type-check   - Run type checker (mypy)"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make test-schema  - Run schema validation tests"
	@echo "  make test-api     - Run API endpoint tests"
	@echo "  make test-stacking - Run stacking ensemble tests"
	@echo "  make validate-data - Validate data files against schema"
	@echo "  make train        - Train default model (CatBoost)"
	@echo "  make train-all    - Train all models"
	@echo "  make oof-all      - Generate OOF for cat,xgb,lgbm (for stacking)"
	@echo "  make stacking     - Run stacking ensemble"
	@echo "  make clean        - Clean generated files"
	@echo "  make check        - Run lint + type-check + test (CI equivalent)"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run API server in Docker"
	@echo "  make docker-test  - Build and smoke-test Docker image"
	@echo "  make mlflow-ui    - Start MLflow UI"

# Installation
install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,api]"
	pre-commit install

# Code Quality
lint:
	ruff check src tests

format:
	ruff check --fix src tests
	ruff format src tests

type-check:
	mypy src

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

test-schema:
	pytest tests/test_schema.py -v

test-api:
	pytest tests/test_api.py -v

test-stacking:
	pytest tests/test_stacking.py -v

# CI equivalent: lint + type-check + test
check: lint type-check test

# Data Validation
validate-data:
	$(PYTHON) -c "import pandas as pd; from src.schema import validate_train_data, validate_test_data; \
		train = pd.read_csv('data/train.csv'); \
		test = pd.read_csv('data/test.csv'); \
		validate_train_data(train); print('✓ Train data valid'); \
		validate_test_data(test); print('✓ Test data valid')"

# Training
train:
	$(PYTHON) -m src.train --model cat --engineered --tune --n-trials 50

train-rf:
	$(PYTHON) -m src.train --model rf --engineered --tune

train-lgbm:
	$(PYTHON) -m src.train --model lgbm --engineered --tune

train-xgb:
	$(PYTHON) -m src.train --model xgb --engineered --tune

train-cat:
	$(PYTHON) -m src.train --model cat --engineered --tune --native-cats

train-all: train-rf train-lgbm train-xgb train-cat

train-kfold:
	$(PYTHON) -m src.train --model cat --engineered --tune --folds 5

# OOF Generation & Stacking
oof-all:
	$(PYTHON) -m src.oof_generator --model cat,xgb,lgbm --engineered --folds 5 --tune --n-trials 50

stacking: oof-all
	$(PYTHON) -m src.stacking --models cat,xgb,lgbm --meta lr

# MLflow
mlflow-ui:
	mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000

# Docker
docker-build:
	docker build -t $(PROJECT_NAME) .

docker-run:
	docker run -p 8000:8000 -v ./output/models:/app/output/models:ro $(PROJECT_NAME)

docker-test: docker-build
	@echo "Smoke-testing Docker image..."
	@docker run -d --name $(PROJECT_NAME)-test -p 8000:8000 $(PROJECT_NAME) > /dev/null
	@sleep 3
	@curl -sf http://localhost:8000/health > /dev/null && echo "✓ /health OK" || echo "✗ /health FAILED"
	@curl -sf http://localhost:8000/model/info > /dev/null && echo "✓ /model/info OK" || echo "✗ /model/info FAILED"
	@docker stop $(PROJECT_NAME)-test > /dev/null && docker rm $(PROJECT_NAME)-test > /dev/null
	@echo "Docker smoke test passed."

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# API
api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Data
download-data:
	kaggle competitions download -c playground-series-s6e2 -p data/
	unzip -o data/playground-series-s6e2.zip -d data/

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf output/models/*.pkl output/submissions/*.csv mlruns/
