# Heart Disease Prediction - MLOps Pipeline

[![CI](https://github.com/white1107/playground-s6e2-mle/actions/workflows/ci.yml/badge.svg)](https://github.com/white1107/playground-s6e2-mle/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-ready ML pipeline for Heart Disease Prediction from [Kaggle Playground Series S6E2](https://www.kaggle.com/competitions/playground-series-s6e2).

## Highlights

- **Best Score**: **0.95395** AUC on Kaggle LB (CatBoost + engineered features)
- **Multiple Models**: CatBoost, LightGBM, XGBoost, Random Forest + stacking ensemble
- **18 Engineered Features**: Domain-knowledge features with schema validation
- **Production API**: FastAPI with 4 endpoints, Pydantic validation, CORS
- **79 Tests**: Comprehensive test suite (API, schema, features, pipeline, stacking)
- **Experiment Tracking**: MLflow integration for reproducible experiments
- **Automated CI/CD**: GitHub Actions with lint, type-check, and tests

## Project Structure

```
playground-s6e2-mle/
├── src/
│   ├── train.py                 # Main training script with MLflow
│   ├── train_baseline.py        # Baseline model (Random Forest)
│   ├── feature_engineering.py   # 18 domain features + original stats
│   ├── schema.py                # Pandera schemas (train/test/submission)
│   ├── stacking.py              # Stacking ensemble with meta-learners
│   ├── ensemble.py              # Model ensembling
│   ├── oof_generator.py         # OOF prediction generator
│   ├── multi_seed.py            # Multi-seed averaging
│   ├── advanced_ensemble.py     # Advanced ensemble methods
│   ├── target_encoding.py       # Target encoding features
│   ├── knn_features.py          # KNN-based features
│   ├── top1_pipeline.py         # Best single-model pipeline
│   ├── shap_analysis.py         # SHAP interpretability
│   ├── correlation_analysis.py  # Feature correlation analysis
│   ├── api/
│   │   └── main.py              # FastAPI (4 endpoints)
│   └── utils/
│       └── mlflow_utils.py      # MLflow tracking utilities
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_api.py              # API endpoint tests (18)
│   ├── test_features.py         # Feature engineering tests (15)
│   ├── test_pipeline.py         # Pipeline tests (9)
│   ├── test_schema.py           # Schema validation tests (25)
│   └── test_stacking.py         # Stacking ensemble tests (12)
├── data/                        # Dataset (not tracked)
├── output/
│   ├── models/                  # Trained models
│   ├── predictions/             # OOF predictions
│   └── submissions/             # Kaggle submissions
├── .github/workflows/ci.yml     # CI/CD pipeline
├── pyproject.toml               # Project configuration
├── Makefile                     # Development commands
├── Dockerfile                   # Container configuration
├── FINDINGS.md                  # Experiment findings & analysis
└── docker-compose.yml
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/white1107/playground-s6e2-mle.git
cd playground-s6e2-mle

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
make install-dev
```

### Training

```bash
# Train CatBoost with feature engineering (default)
make train

# Train specific model
python -m src.train --model cat --engineered --tune --n-trials 50

# Generate OOF predictions & stacking ensemble
make stacking

# Available models: rf, lgbm, xgb, cat
```

### MLflow Experiment Tracking

```bash
# Start MLflow UI
make mlflow-ui

# Open http://localhost:5000 in browser
```

### API Server

```bash
# Run locally
make api

# Run with Docker
make docker-build
make docker-run
```

## Model Performance

| Model | Features | Tuning | CV AUC | LB Score |
|:------|:---------|:-------|:-------|:---------|
| **CatBoost** | Engineered | Optuna | **0.95549** | **0.95395** |
| XGBoost | Engineered | Optuna | 0.95530 | 0.95351 |
| CatBoost (raw 13) | Raw | 10-seed | 0.95537 | 0.95341 |
| Blend (0.8 eng + 0.2 raw) | Mixed | - | 0.95557 | 0.95321 |
| CatBoost (10-seed) | Engineered | Optuna | 0.95556 | 0.95292 |
| RealMLP | Engineered | - | 0.95566 | 0.94639 |
| Baseline RF | Numeric only | Default | - | 0.85498 |

**Key Insight**: Single CatBoost with Optuna outperforms all complex ensembles on LB. Multi-seed averaging and stacking hurt due to model homogeneity (Spearman > 0.997).

## Feature Engineering

### Base Features (13)

**Numerical (6)**: Age, BP, Cholesterol, Max HR, ST depression, Number of vessels fluro

**Categorical (7)**: Sex, Chest pain type, FBS over 120, EKG results, Exercise angina, Slope of ST, Thallium

### Engineered Features (18)

| Feature | Formula | Description |
|:--------|:--------|:------------|
| Rate_Pressure_Product | BP x Max HR | Myocardial oxygen demand |
| Electrical_Stress | ST depression x Slope of ST | Cardiac electrical stress |
| Metabolic_Score | Sum of risk factors | Metabolic risk composite (0-3) |
| MaxHR_Rel_Age | Max HR / (220 - Age) | Heart rate reserve ratio |
| MaxHR_x_Age | Max HR x Age | Age-adjusted heart capacity |
| BP_x_Cholesterol | BP x Cholesterol | Combined cardiovascular risk |
| Age_Bin | Age // 10 | Decade-based age grouping |
| Cholesterol_per_Age | Cholesterol / Age | Cholesterol burden |
| BP_per_Age | BP / Age | Hypertension severity |
| HR_Deficit | (220 - Age) - Max HR | Heart rate deficit |
| Exercise_Risk | Angina*2 + ST + slope flag | Exercise risk composite |
| Vessel_Thallium | Vessels x Thallium | Diagnostic interaction |
| Angina_ST | Angina x ST depression | Angina-ST interaction |
| Cardiac_Risk | Framingham-inspired score | Composite risk (0-5) |
| ST_per_HR | ST depression / Max HR | ST normalized by effort |
| Typical_Angina | Chest pain type == 4 | Typical angina flag |
| Has_Vessel | Vessels > 0 | Vessel involvement flag |
| Thallium_Abnormal | Thallium != 3 | Abnormal thallium flag |

## API Usage

### Endpoints

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET` | `/health` | Health check (status, model_loaded, version) |
| `POST` | `/predict` | Single prediction with risk level |
| `POST` | `/predict/batch` | Batch prediction with timing |
| `GET` | `/model/info` | Model metadata and best scores |

### Examples

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 63, "Sex": 1, "Chest pain type": 4,
    "BP": 145, "Cholesterol": 233, "FBS over 120": 1,
    "EKG results": 0, "Max HR": 150, "Exercise angina": 0,
    "ST depression": 2.3, "Slope of ST": 3,
    "Number of vessels fluro": 0, "Thallium": 6
  }'
# → {"probability": 0.82, "prediction": 1, "risk_level": "High"}

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[{"Age": 45, ...}, {"Age": 60, ...}]'
# → {"predictions": [...], "count": 2, "processing_time_ms": 12.5}

# Model info
curl http://localhost:8000/model/info
# → {"model_type": "CatBoost", "features": [...], "best_cv_auc": 0.95549, ...}
```

## Testing

```bash
# Run all tests (79 tests)
make test

# Run specific test suites
make test-api        # API endpoint tests
make test-stacking   # Stacking ensemble tests
make test-schema     # Schema validation tests

# Run with coverage
make test-cov
```

## MLOps Features

### CI/CD Pipeline

GitHub Actions workflow includes:
- **Lint & Format**: ruff check and format
- **Type Check**: mypy static analysis
- **Tests**: pytest with coverage (79 tests)
- **Build**: Package build verification
- **Docker**: Container build test

### Data Validation (Pandera)

Schemas for train/test/submission data with automatic type coercion and range checks.

### Hyperparameter Tuning (Optuna)

```bash
python -m src.train --model cat --tune --n-trials 100
```

## Development

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

### Code Quality

```bash
make lint        # ruff check
make format      # ruff format
make type-check  # mypy
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Kaggle Playground Series S6E2](https://www.kaggle.com/competitions/playground-series-s6e2)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
