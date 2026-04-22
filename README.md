# ML Pipeline & n8n Setup

## 1. Start n8n (Docker)

```bash
docker volume create n8n_data

docker run -d --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n
```

Open http://localhost:5678 → create account → import `ml_automsation.json` → toggle **Active**.

## 2. Start ML Pipeline API (port 8001)

**Linux/macOS:**
```bash
cd agent
uvicorn ml_pipeline_api.main:app --port 8001 --reload
```

**Windows:**
```bat
cd agent
start_ml_pipeline.bat
```

Endpoints:
- `POST http://127.0.0.1:8001/run` — upload CSV → churn + segmentation → save to DB
- `GET http://127.0.0.1:8001/status` — last run info
- `GET http://127.0.0.1:8001/health` — health check

## 3. Start Agent Backend (port 8000)

**Linux/macOS:**
```bash
cd agent
uvicorn backend.main:app --port 8000 --reload
```

**Windows:**
```bat
cd agent
start_backend.bat
```

## 4. Start Frontend (port 4200)

**Linux/macOS:**
```bash
cd frontend/artisan-materials
npm start
```

**Windows:**
```bat
cd frontend
start_frontend.bat
```

## Architecture

```
n8n (port 5678)
  └─► POST http://127.0.0.1:8001/run  (CSV → churn + segmentation)

FastAPI Agent   (port 8000) — CRUD + chat + ML prediction
FastAPI ML API  (port 8001) — n8n churn pipeline
Angular         (port 4200) — UI
```

## Test Manually

```bash
# ML Pipeline
curl -X POST http://127.0.0.1:8001/run -F "file=@data/FactVentee.csv"

# Check DB
python3 -c "import sqlite3; c=sqlite3.connect('agent/backend/materials.db'); print('Predictions:', c.execute('SELECT COUNT(*) FROM customer_predictions').fetchone()[0]); c.close()"
```
