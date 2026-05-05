# Koudos ML Integration

Production-like MLOps solution with monitoring, drift detection, and alerting.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  ml-integration/                                            │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  API:8000    │  │  Churn:8001  │  │  Flask Web:3001  │  │
│  │  (FastAPI)   │  │  (FastAPI)   │  │  (Flask)         │  │
│  │              │  │              │  │                  │  │
│  │  - supplier  │  │  - predict   │  │  - dashboards   │  │
│  │  - sell      │  │  - batch     │  │  - forms        │  │
│  │  - promote   │  │  - train     │  │                  │  │
│  │  - campaign  │  │  - drift     │  │                  │  │
│  │  - sougui    │  │              │  │                  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────┘  │
│         │                 │                                  │
│         ▼                 ▼                                  │
│  ┌──────────────────────────────────┐                       │
│  │  Prometheus:9090                 │                       │
│  │  scrapes :8000 + :8001 /15s     │                       │
│  └──────────────┬───────────────────┘                       │
│                 │                                             │
│                 ▼                                             │
│  ┌──────────────────────────────────┐                       │
│  │  Grafana:3000                    │                       │
│  │  dashboards + alerts             │                       │
│  └──────────────────────────────────┘                       │
│                                                              │
│  ┌──────────────────────────────────┐                       │
│  │  MLflow:5000                     │                       │
│  │  experiment tracking             │                       │
│  └──────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Docker + Docker Compose
- PostgreSQL database (external)
- Python 3.11+ (for local development)

## Quick Start

### 1. Configure Database

Copy `.env.example` to `.env` and fill in your PostgreSQL credentials:

```bash
PGHOST=<your-postgres-host>
PGPORT=5432
PGDATABASE=<your-database>
PGUSER=<your-user>
PGPASSWORD=<your-password>
```

### 2. Start All Services

```bash
cd ml-integration
docker compose up --build -d
```

This starts 6 services:

| Service     | Host URL                  | Purpose                        |
|-------------|---------------------------|--------------------------------|
| api         | http://localhost:8000     | Main ML API                    |
| churn-api   | http://localhost:8001     | Churn prediction API           |
| web         | http://localhost:3001     | Flask web dashboard            |
| prometheus  | http://localhost:9090     | Metrics collection             |
| grafana     | http://localhost:3000     | Dashboards + alerting          |
| mlflow      | http://localhost:5000     | Experiment tracking            |

### 3. Verify Everything Is Running

```bash
# Health checks
curl http://localhost:8000/health
# → {"status":"ok"}

curl http://localhost:8001/health
# → {"status":"ok","service":"churn-prediction"}

# Metrics endpoints
curl http://localhost:8000/metrics | head -5
curl http://localhost:8001/metrics | head -5

# Model status
curl http://localhost:8000/models
```

## API Endpoints

### Main API (port 8000) — Swagger docs: http://localhost:8000/docs

#### Supplier Clustering

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/train/supplier/dataset` | View prepared supplier dataset |
| `POST` | `/train/supplier/train` | Train supplier clustering model |
| `POST` | `/predict/supplier` | Predict cluster for a supplier |
| `GET` | `/drift/supplier` | **Check supplier model drift** |
| `POST` | `/lookup/supplier-clusters/save` | Save clusters to PostgreSQL |

**Predict request:**
```bash
curl -X POST http://localhost:8000/predict/supplier \
  -H "Content-Type: application/json" \
  -d '{"quantity": 100, "unit_price": 5.5, "total_ht": 550, "total_ttc": 660, "governorate": "GS", "city": "Tunis"}'
```

**Drift check:**
```bash
curl http://localhost:8000/drift/supplier
# → {"drift_detected": false, "distribution_shift": 0.02, "precision_drop_ratio": 0.01, ...}
```

#### Sell Forecasting

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/train/sell/dataset` | View prepared sell dataset |
| `POST` | `/train/sell/train` | Train sell forecasting model |
| `GET` | `/predict/sell` | Get sell forecast + recommendation |
| `GET` | `/drift/sell` | **Check sell model drift** |
| `POST` | `/lookup/sell-monthly/save` | Save monthly forecast to PostgreSQL |

#### Promote Forecasting

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/train/promote/dataset` | View prepared promote dataset |
| `POST` | `/train/promote/train` | Train promote forecasting model |
| `GET` | `/predict/promote` | Get promote forecast + plan |
| `POST` | `/predict/promote/form` | Predict from form input |
| `GET` | `/drift/promote` | **Check promote model drift** |
| `POST` | `/lookup/promote-monthly/save` | Save monthly plan to PostgreSQL |

#### Campaign Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/train/campaign/dataset` | View prepared campaign dataset |
| `POST` | `/train/campaign/train` | Train campaign models |
| `GET` | `/predict/campaign` | Get campaign prediction |
| `POST` | `/predict/campaign/form` | Predict from form input |

#### Sougui

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sougui/predict` | Sougui classification prediction |

#### Common

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/models` | Model artifact status |
| `GET` | `/metrics` | **Prometheus metrics** |

### Churn API (port 8001) — Swagger docs: http://localhost:8001/docs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Predict churn for single customer |
| `POST` | `/predict/batch` | Batch predict for multiple customers |
| `POST` | `/predict/batch/csv` | Batch predict from CSV upload |
| `POST` | `/predict/batch/transactions` | Derive features + predict from raw transactions |
| `POST` | `/predict/train` | Train churn pipeline |
| `GET` | `/predict/pca-plot` | PCA segmentation plot (PNG) |
| `GET` | `/predict/segment-labels` | Segment labels and profiles |
| `GET` | `/predict/models` | Model artifact status |
| `GET` | `/predict/status` | Service readiness |
| `GET` | `/health` | Health check |
| `GET` | `/train/status` | Training pipeline status |
| `GET` | `/metrics` | **Prometheus metrics** |
| `GET` | `/drift/churn` | **Check churn model drift** |

**Single prediction:**
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "recency": 30,
    "frequency": 5,
    "monetary_total": 1500,
    "monetary_trend": 0.1,
    "product_diversity": 3,
    "channel_diversity": 2,
    "lifetime_days": 365,
    "purchase_velocity": 0.5,
    "avg_price": 300,
    "channel": "online"
  }'
```

**Drift check:**
```bash
curl http://localhost:8001/drift/churn
# → {"drift_detected": false, "avg_probability_shift": 0.02, "high_risk_rate": 0.15, ...}
```

## Monitoring

### Grafana Dashboard

1. Open **http://localhost:3000**
2. Login: `admin` / `admin`
3. Add Prometheus data source:
   - URL: `http://prometheus:9090`
   - Save & Test
4. Import dashboard:
   - Dashboard → Import → Upload JSON file
   - Select `monitoring/grafana_dashboard.json`

The dashboard has **11 panels**:

| Row | Left Panel | Right Panel |
|-----|------------|-------------|
| 1 | Traffic by endpoint | Latency p95 by endpoint |
| 2 | Error rate | Model health — supplier |
| 3 | Data health (full width) | — |
| 4 | Model health — sell / promote | Model health — churn |
| 5 | Drift — distribution shift | Drift — precision / accuracy drop |
| 6 | Drift — detected flag | Retraining triggers |

### Prometheus

- UI: **http://localhost:9090**
- Query `ml_model_metric_value` to see all model metrics
- Query `ml_drift_metric_value` to see drift metrics
- Rules tab: view configured alerts

### Alert Rules

7 alert rules are configured in `monitoring/alerts.yml`:

| Alert | Condition | Severity |
|-------|-----------|----------|
| `HighLatency` | p95 latency > 2s for 5m | warning |
| `HighErrorRate` | 5xx rate > 5% for 5m | critical |
| `ModelPrecisionDrop` | supplier precision drop > 5% | warning |
| `DriftDetected` | supplier drift flagged | critical |
| `SellAccuracyDrop` | sell accuracy drop > 5% | warning |
| `SellDriftDetected` | sell drift flagged | critical |
| `PromoteAccuracyDrop` | promote accuracy drop > 5% | warning |
| `PromoteDriftDetected` | promote drift flagged | critical |
| `ChurnAccuracyDrop` | churn accuracy drop > 5% | warning |
| `ChurnDriftDetected` | churn drift flagged | critical |
| `HighChurnRiskRate` | high-risk rate > 30% for 10m | warning |

## Simulation Scenarios

All three scripts target the **churn API** (port 8001). After running each, check Grafana to see the impact.

### 1. High Traffic → Observe Latency Impact

Sends 500 concurrent requests (20 at a time) to the churn API (`/health` + `/predict`). Watch the **Traffic** and **Latency p95** panels in Grafana.

```bash
pip install httpx
python3 simulate_high_traffic.py --count 500 --concurrency 20
```

### 2. API Errors → Observe Error Spikes

Sends 100 malformed requests to churn prediction endpoints (`/predict`, `/predict/batch`) with invalid data types. Watch the **Error rate** panel in Grafana.

```bash
pip install httpx
python3 simulate_errors.py --count 100
```

### 3. Model Drift → Observe Degradation

First makes valid predictions to establish a baseline, then hits `/drift/churn` with drifted data. Watch the **Drift** and **Model health — churn** panels in Grafana.

```bash
pip install httpx
python3 simulate_drift.py --count 50
```

### Expected Grafana Results After Running All Three

| Panel | Shows After |
|-------|-------------|
| Traffic by endpoint | Spike from `simulate_high_traffic.py` |
| Latency p95 | Latency impact from concurrent requests |
| Error rate | Error spike from `simulate_errors.py` |
| Model health — churn | avg_churn_probability + high_risk_rate from predictions |
| Drift — detected flag | Drift status from `simulate_drift.py` |
| Retraining triggers | Count of retraining events |

## Development

### Local Python Setup

```bash
cd ml-integration
pip install -r requirements.txt
pip install -r requirements-churn.txt

# Run main API locally
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run churn API locally (separate terminal)
uvicorn churn_api:app --host 0.0.0.0 --port 8001 --reload
```

### Add a New Model with Drift Detection

1. Create model API module (e.g., `new_model_api.py`)
2. Import observability functions:
   ```python
   from observability import (
       log_event, publish_data_metrics, publish_drift_metrics,
       publish_model_metrics, record_retraining_trigger,
   )
   ```
3. Define model name constant:
   ```python
   MODEL_NAME = "my_new_model"
   ```
4. In training endpoint, publish baseline metrics:
   ```python
   publish_model_metrics(MODEL_NAME, {"mae": mae, "rmse": rmse})
   publish_drift_metrics(MODEL_NAME, {"accuracy_drop_ratio": 0.0, "detected": 0.0})
   record_retraining_trigger(MODEL_NAME, "manual_train")
   ```
5. Add drift endpoint:
   ```python
   @router.get("/drift/my_model", response_model=MyDriftResponse)
   def detect_drift() -> MyDriftResponse:
       # compare current metrics vs baseline
       # publish_drift_metrics(MODEL_NAME, {"accuracy_drop_ratio": ratio, "detected": 1.0})
   ```
6. Add alert rule in `monitoring/alerts.yml`:
   ```yaml
   - alert: MyModelDriftDetected
     expr: ml_drift_metric_value{model="my_new_model",metric="detected"} == 1
     for: 5m
     labels:
       severity: critical
   ```
7. Add panel in `monitoring/grafana_dashboard.json`

## Troubleshooting

### Services won't start

```bash
# Check logs
docker compose logs api
docker compose logs churn-api

# Check .env file
cat .env
```

### Prometheus not scraping

```bash
# Check targets
curl http://localhost:9090/api/v1/targets | python3 -m json.tool

# Verify host-gateway resolution
docker compose exec prometheus cat /etc/hosts | grep localhost
```

### Grafana no data

1. Verify Prometheus data source is configured
2. Check Prometheus has data: `curl http://localhost:9090/api/v1/query?query=up`
3. Import dashboard JSON again

### Port conflicts

| Port | Default | Alternative |
|------|---------|-------------|
| Grafana | 3000 | Edit docker-compose.yml |
| Web | 3001 | Edit docker-compose.yml |
| API | 8000 | Edit docker-compose.yml |
| Churn | 8001 | Edit docker-compose.yml |
| Prometheus | 9090 | Edit docker-compose.yml |
| MLflow | 5000 | Edit docker-compose.yml |

## File Structure

```
ml-integration/
├── main.py                      # Main FastAPI app (supplier, sell, promote, campaign, sougui)
├── churn_api.py                 # Churn FastAPI app (separate service, port 8001)
├── observability.py             # Shared metrics + logging (Prometheus-compatible)
├── supplier_api.py              # Supplier clustering API + drift detection
├── sell_api.py                  # Sell forecasting API + drift detection
├── promote_api.py               # Promote forecasting API + drift detection
├── campaign_api.py              # Campaign analytics API
├── sougui_api.py                # Sougui classification API
├── churn_pipeline.py            # Churn inference pipeline
├── train_pipeline.py            # Churn training pipeline
├── flask_app.py                 # Flask web dashboard
├── docker-compose.yml           # All services (api, churn-api, web, prometheus, grafana, mlflow)
├── Dockerfile                   # Main API + web + mlflow
├── Dockerfile.churn             # Churn API
├── requirements.txt             # Main API dependencies
├── requirements-churn.txt       # Churn API dependencies
├── monitoring/
│   ├── prometheus.yml           # Prometheus scrape config (api:8000 + churn-api:8001)
│   ├── alerts.yml               # Alert rules (latency, errors, drift, churn risk)
│   └── grafana_dashboard.json   # Grafana dashboard (11 panels)
├── models/                      # Model artifacts (.pkl files)
├── churn_artifacts/             # Churn model artifacts
├── templates/                   # Flask HTML templates
├── static/                      # Flask static files (CSS, JS)
└── sougui_mlops/                # Sougui ML submodule
```
