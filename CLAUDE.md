# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/

# Run single test file
pytest tests/test_api.py

# Run single test function
pytest tests/test_api.py::test_decide_thresholds

# Lint
ruff check .

# Start FastAPI dev server
uvicorn app.main:app --reload --port 8000

# Start full stack (FastAPI + MLflow)
docker-compose up

# Train model (not yet implemented)
python -m src.models.train
```

## Architecture

**Two-layer design:** ML pipeline (`src/`) feeds a FastAPI service (`app/`).

### ML Pipeline (`src/`)

```
Raw CSVs (data/raw/) → src/data/features.py → src/data/preprocessing.py → src/models/train.py → MLflow registry
```

- `src/data/features.py` — adds derived features (debt-to-income ratio, credit utilization, etc.) **before** the sklearn pipeline
- `src/data/preprocessing.py` — builds a `ColumnTransformer`/`Pipeline` for numeric imputation+scaling and categorical imputation+encoding; must fit only on training data
- `src/models/train.py` — orchestrates training (Logistic Regression → RF → XGBoost/LightGBM), Optuna tuning, MLflow logging, model registration
- `src/models/evaluate.py` — imbalanced-aware metrics: ROC-AUC, PR-AUC, F1; threshold tuning for cost-sensitive decisions
- `src/explainability/shap_explainer.py` — global (beeswarm/feature importance) + local (waterfall) SHAP explanations

### API Service (`app/`)

```
POST /predict → app/predict.py (business rules: PD → decision + risk tier + rate)
POST /explain → app/explain.py → src/explainability/shap_explainer.py
```

- Business rule thresholds in `app/predict.py`: PD < 0.15 → `approve`, 0.15–0.40 → `manual_review`, > 0.40 → `reject`
- `app/schemas.py` — Pydantic v2 models: `PredictRequest` (fields TBD), `PredictResponse`, `ExplainResponse`
- MLflow tracking URI injected via env var `MLFLOW_TRACKING_URI` (set to `http://mlflow:5000` in docker-compose)

### Experiment Tracking

MLflow runs on port 5000 (see `docker-compose.yml`). Artifacts stored under `./mlflow/` (git-ignored). Model registry used to version best model served by the API.

## Dataset

Home Credit Default Risk (Kaggle). Primary table: `data/raw/application_train.csv` (~307k rows, 120+ features, ~8% default rate). Supplementary tables: `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv`.

## Implementation Status

All core functions are stubs (`raise NotImplementedError`). Only implemented:
- `app/predict.py::decide()` — business rule thresholds
- `tests/test_api.py::test_decide_thresholds` — one passing test

Notebooks (`notebooks/01–04`) are the intended development scratch space before promoting code to `src/`.
