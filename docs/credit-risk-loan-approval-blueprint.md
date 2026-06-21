# Credit Risk & Loan Approval System — Project Blueprint

> **One-liner:** Một end-to-end ML system dự đoán xác suất vỡ nợ (Probability of Default), đóng gói thành Loan Approval API có khả năng giải thích quyết định bằng SHAP — từ raw data đến production-ready service với CI/CD pipeline.

---

## 1. Bối cảnh & Business Value

### Tại sao bài toán này tồn tại?

Trong quy trình cho vay thực tế tại ngân hàng, quyết định approve/reject một hồ sơ không đến từ một model "Loan Approval" đơn thuần. Nó đến từ một pipeline:

```
Hồ sơ vay
    ↓
Credit Risk Model  →  Probability of Default (PD)
    ↓
Business rules     →  PD < 0.15 → Approve
                       0.15 ≤ PD ≤ 0.40 → Manual review
                       PD > 0.40 → Reject
    ↓
SHAP Explanation   →  "Tại sao hồ sơ này bị reject?"
    ↓
Quyết định cuối
```

Credit Risk model **là engine bên trong** của Loan Approval system. Loan Approval chỉ là business logic layer đặt lên trên output của Credit Risk.

### Business value thực tế

- **Risk-based pricing:** Khách hàng PD = 0.05 và PD = 0.18 đều được approve, nhưng với lãi suất khác nhau — một quyết định binary approve/reject không làm được điều này.
- **Portfolio management:** Ngân hàng tính toán expected loss toàn bộ danh mục — yêu cầu bắt buộc theo chuẩn Basel III/IV mà các ngân hàng VN đang tuân thủ.
- **Regulatory explainability:** Quyết định tín dụng phải giải thích được. SHAP values đáp ứng yêu cầu này.

---

## 2. Kiến trúc tổng thể

```
Raw Data (Kaggle)
    ↓
[EDA + Feature Engineering]  ←── Pandas, Seaborn, Scikit-learn Pipeline
    ↓
[Model Training]             ←── XGBoost / LightGBM, cross-validation, imbalanced handling
    ↓
[Experiment Tracking]        ←── MLflow (params, metrics, artifacts, model registry)
    ↓
[Model Serving]              ←── FastAPI (/predict + /explain endpoints)
    ↓
[Containerization]           ←── Docker + Docker Compose
    ↓
[CI/CD]                      ←── GitHub Actions (pytest → build → push image → deploy)
    ↓
[Production]                 ←── Render / Railway / EC2 free tier
```

**Optional (nếu muốn showcase Data Engineering):**
```
Raw Data → dbt (SQL transforms) → Airflow DAG → Feature store → Model Training
```

---

## 3. Dataset

**Primary recommendation:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
- ~307,000 hồ sơ vay, 120+ features
- Imbalanced: ~8% default — buộc bạn phải xử lý imbalanced data
- Nhiều bảng dữ liệu liên quan (application, bureau, previous application) — feature engineering phong phú

**Alternative:** [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit)
- Nhỏ hơn, dễ bắt đầu hơn (~150k rows, 10 features)
- Phù hợp nếu muốn tập trung vào pipeline hơn là feature engineering

---

## 4. Các bước triển khai

### Bước 1 — EDA & Data Understanding
- Phân tích phân phối target (default rate)
- Missing value analysis, outlier detection
- Correlation analysis giữa features và target
- Visualize class imbalance

**Output:** EDA notebook với insights rõ ràng, có thể dùng làm story trong README.

### Bước 2 — Feature Engineering & Preprocessing
- Xử lý missing values (imputation strategies khác nhau cho numeric vs categorical)
- Encoding: Target encoding / One-hot encoding
- Feature scaling
- Tạo derived features (debt-to-income ratio, credit utilization rate...)
- Build Scikit-learn `Pipeline` + `ColumnTransformer` để reproducible

**Key concept:** Toàn bộ preprocessing phải nằm trong một Sklearn Pipeline — không được fit trên test set (data leakage).

### Bước 3 — Xử lý Imbalanced Data
- Baseline: class_weight='balanced'
- SMOTE (oversampling minority class)
- Threshold tuning: không dùng default 0.5, tune threshold để optimize F1 hoặc business metric (cost-sensitive)
- Evaluation metrics đúng: không dùng Accuracy, dùng ROC-AUC, PR-AUC, F1

### Bước 4 — Model Training & Experiment Tracking
- Train nhiều models: Logistic Regression (baseline) → Random Forest → XGBoost → LightGBM
- Hyperparameter tuning: Optuna hoặc GridSearchCV
- Log tất cả vào **MLflow**: params, metrics, model artifacts
- Register model tốt nhất vào MLflow Model Registry
- Promote model qua stages: Staging → Production

### Bước 5 — SHAP Explainability
- Global explanation: feature importance dựa trên SHAP values (thay thế cho feature_importances_ mặc định)
- Local explanation: với mỗi prediction, tính SHAP values để giải thích "tại sao hồ sơ này bị reject"
- Visualize: waterfall plot, beeswarm plot, dependence plot
- Tích hợp vào API: endpoint `/explain/{application_id}` trả về top features ảnh hưởng đến quyết định

### Bước 6 — FastAPI Service
```
POST /predict
  Input:  { application features }
  Output: {
    "probability_of_default": 0.23,
    "decision": "manual_review",        ← business rule layer
    "risk_tier": "medium",
    "recommended_interest_rate": "12.5%"  ← risk-based pricing demo
  }

POST /explain
  Input:  { application features }
  Output: {
    "top_factors": [
      { "feature": "debt_to_income_ratio", "impact": +0.18, "direction": "increases_risk" },
      { "feature": "credit_history_length", "impact": -0.12, "direction": "decreases_risk" },
      ...
    ],
    "summary": "Hồ sơ này có rủi ro cao chủ yếu do tỷ lệ nợ/thu nhập cao."
  }
```

### Bước 7 — Docker & Containerization
```dockerfile
# Dockerfile skeleton
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- `docker-compose.yml` bao gồm: FastAPI service + MLflow tracking server
- Mount model artifacts từ MLflow registry vào container

### Bước 8 — CI/CD với GitHub Actions

```yaml
# .github/workflows/ci.yml (skeleton)
on: [push]
jobs:
  test:
    steps:
      - run: pytest tests/          # unit tests cho preprocessing + model inference
  build:
    needs: test
    steps:
      - run: docker build ...       # build image
      - run: docker push ...        # push to Docker Hub / GHCR
  deploy:
    needs: build
    steps:
      - ...                         # trigger deploy trên Render/Railway
```

**Tests cần viết:**
- Test preprocessing pipeline (đầu vào → đầu ra đúng shape/dtype)
- Test model inference (prediction nằm trong [0, 1])
- Test business logic layer (threshold → decision mapping đúng)
- Test API endpoints (status codes, response schema)

---

## 5. Bộ Techstack & Skills

### Core ML
| Tool | Mục đích |
|---|---|
| Python 3.11 | Ngôn ngữ chính |
| Pandas / Polars | Data manipulation |
| Scikit-learn | Preprocessing pipeline, baseline models, metrics |
| XGBoost / LightGBM | Main models |
| Imbalanced-learn | SMOTE, resampling |
| Optuna | Hyperparameter tuning |
| SHAP | Explainability |

### Experiment & Tracking
| Tool | Mục đích |
|---|---|
| MLflow | Experiment tracking, model registry, artifact store |
| Matplotlib / Seaborn | EDA visualization |
| Plotly | Interactive charts (optional) |

### Serving & Infra
| Tool | Mục đích |
|---|---|
| FastAPI | REST API (bạn đã có kinh nghiệm) |
| Pydantic v2 | Request/response validation |
| Uvicorn | ASGI server |
| Docker | Containerization |
| Docker Compose | Local multi-service orchestration |

### CI/CD
| Tool | Mục đích |
|---|---|
| GitHub Actions | CI/CD pipeline |
| pytest | Unit & integration testing |
| Ruff / Black | Linting & formatting |

### Optional — Data Engineering
| Tool | Mục đích |
|---|---|
| Apache Airflow | DAG orchestration cho training pipeline |
| dbt | SQL-based feature transformation |

### Skills được showcase
- **EDA & data storytelling** — biến data thành insights có thể giải thích
- **Imbalanced classification** — xử lý thực tế, không phải textbook
- **ML pipeline design** — reproducible, no data leakage
- **Experiment tracking** — tư duy về reproducibility và model versioning
- **Model explainability (SHAP)** — kỹ năng ít fresher có, relevant với Finance domain
- **API design** — FastAPI, Pydantic schema design
- **Containerization** — Docker, biết cách đóng gói một ML service
- **CI/CD** — GitHub Actions, automated testing
- **Business framing** — biết cách map ML output sang business decision

---

## 6. Differentiators so với project thông thường

Hầu hết intern portfolio dừng lại ở bước "train model → notebook". Project này vượt xa hơn ở các điểm:

1. **SHAP explainability** — không chỉ predict mà còn explain. Đây là yêu cầu thực tế trong Finance mà ít người làm.
2. **Business logic layer** — thêm risk-based pricing và decision tiers, biến ML output thành business output.
3. **Production-ready API** — không phải Streamlit demo, mà là REST API đúng nghĩa với schema validation.
4. **CI/CD pipeline** — code push → test → build → deploy tự động.
5. **MLflow model registry** — tư duy về model lifecycle, không chỉ "train và save pkl".

---

## 7. Suggested Repository Structure

```
credit-risk-loan-approval/
├── data/
│   ├── raw/                    # raw data (gitignored)
│   └── processed/              # processed features
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_explainability.ipynb
├── src/
│   ├── data/
│   │   ├── preprocessing.py    # Sklearn Pipeline
│   │   └── features.py         # Feature engineering logic
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── explainability/
│       └── shap_explainer.py
├── app/
│   ├── main.py                 # FastAPI app
│   ├── schemas.py              # Pydantic models
│   ├── predict.py              # inference logic
│   └── explain.py              # SHAP explanation logic
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── mlflow/                     # MLflow tracking URI config
└── README.md                   # Project story + demo
```

---

## 8. README Story (gợi ý narrative)

README không nên chỉ là "how to run". Nên kể một câu chuyện:

1. **Problem** — Ngân hàng cần đưa ra quyết định tín dụng nhanh, chính xác, và có thể giải thích được theo yêu cầu pháp lý.
2. **Approach** — Xây dựng Credit Risk model (PD) + business rule layer + SHAP explainer.
3. **Results** — ROC-AUC đạt X, PR-AUC đạt Y, model explain được top 5 factors cho mỗi quyết định.
4. **Demo** — Link đến deployed API endpoint + example request/response.
5. **Architecture diagram** — pipeline từ data đến production.

---

*Blueprint version 1.0 — generated from project planning discussion*
