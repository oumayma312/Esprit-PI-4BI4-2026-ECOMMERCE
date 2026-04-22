"""Separate FastAPI service for n8n ML automation pipeline.

Runs on port 8001, separate from the agent backend (port 8000).
Shares the same SQLite DB and ml_pipeline.py module.

Usage:
    uvicorn ml_pipeline_api.main:app --host 0.0.0.0 --port 8001
"""
import logging
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add parent directories to path for shared modules
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from ml_pipeline import get_pipeline
from backend.db import get_db

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = str(_ROOT / "artifacts")
DB_PATH = str(Path(__file__).resolve().parent.parent / "backend" / "materials.db")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ml_pipeline_api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ML Churn Pipeline",
    description="n8n automation pipeline for churn prediction + segmentation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/run")
def api_run_churn_pipeline():
    """Full ML churn pipeline: read CSV from known path, derive features, predict, save to DB."""
    logger.info("ML churn pipeline triggered")
    start = time.time()

    pipeline = get_pipeline(ARTIFACTS_DIR)

    # Read CSV from known path (project root / data /)
    csv_path = _ROOT / "data" / "FactVentee.csv"
    try:
        tx_df = pd.read_csv(csv_path)
        logger.info(f"CSV loaded: {len(tx_df)} rows, {len(tx_df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise HTTPException(500, f"Failed to read CSV: {e}")

    # Derive features from raw transactions
    try:
        features_df, derivation_issues = pipeline.derive_features_from_transactions(tx_df)
        logger.info(f"Features derived: {len(features_df)} customers, issues: {derivation_issues}")
    except Exception as e:
        logger.error(f"Feature derivation failed: {e}")
        raise HTTPException(500, f"Feature derivation failed: {e}")

    # Run churn prediction + segmentation
    try:
        results_df, inference_issues = pipeline.run_inference(features_df, "")
        all_issues = derivation_issues + inference_issues
        logger.info(f"Inference complete: {len(results_df)} predictions")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(500, f"Inference failed: {e}")

    # Save predictions to DB (match by customerFK as code/name)
    saved = 0
    failed = 0
    for _, row in results_df.iterrows():
        customer_name = str(row.get("customerFK", ""))
        try:
            with get_db() as conn:
                cust = conn.execute(
                    "SELECT id FROM customers WHERE code = ? OR name = ?",
                    (customer_name, customer_name),
                ).fetchone()

                if not cust:
                    cur = conn.execute(
                        "INSERT INTO customers (name, code, active) VALUES (?, ?, 1)",
                        (customer_name, customer_name),
                    )
                    customer_id = cur.lastrowid
                else:
                    customer_id = cust["id"]

                # Inline SQL — no nested get_db() call
                conn.execute("""
                    INSERT INTO customer_predictions
                        (customer_id, churn_probability, churn_prediction, segment, segment_label, risk_level, recommendation)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(customer_id) DO UPDATE SET
                        churn_probability = excluded.churn_probability,
                        churn_prediction = excluded.churn_prediction,
                        segment = excluded.segment,
                        segment_label = excluded.segment_label,
                        risk_level = excluded.risk_level,
                        recommendation = excluded.recommendation,
                        predicted_at = datetime('now')
                """, (customer_id, float(row["churn_probability"]), int(row["churn_prediction"]),
                      int(row["segment"]), str(row["segment_label"]), str(row["risk_level"]),
                      str(row["recommendation"])))
                saved += 1
        except Exception as e:
            logger.warning(f"Failed to save prediction for {customer_name}: {e}")
            failed += 1

    # Save transactions to DB
    try:
        for _, tx_row in tx_df.iterrows():
            customer_name = str(tx_row.get("customerFK", ""))
            with get_db() as conn:
                cust = conn.execute(
                    "SELECT id FROM customers WHERE code = ? OR name = ?",
                    (customer_name, customer_name),
                ).fetchone()
                if not cust:
                    cur = conn.execute(
                        "INSERT INTO customers (name, code, active) VALUES (?, ?, 1)",
                        (customer_name, customer_name),
                    )
                    customer_id = cur.lastrowid
                else:
                    customer_id = cust["id"]

                date_val = tx_row.get("dateFK", "")
                if isinstance(date_val, (int, float)):
                    date_val = str(int(date_val))
                date_str = str(date_val) if pd.notna(date_val) else ""

                # Inline SQL — no nested get_db() call
                conn.execute("""
                    INSERT INTO customer_transactions (customer_id, transaction_date, total_ttc, quantity, price, channelFK, productFK)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (customer_id, date_str, float(tx_row.get("total_ttc", 0)),
                      int(tx_row.get("quantity", 0)), float(tx_row.get("price", 0)),
                      int(tx_row.get("channelFK", 0)) if pd.notna(tx_row.get("channelFK")) else None,
                      int(tx_row.get("productFK", 0)) if pd.notna(tx_row.get("productFK")) else None))
    except Exception as e:
        logger.warning(f"Transaction save failed (non-critical): {e}")

    elapsed = time.time() - start
    logger.info(f"ML churn pipeline complete: {saved} saved, {failed} failed, {elapsed:.2f}s")

    high_risk = int((results_df["churn_probability"] >= 0.7).sum())
    mod_risk = int(((results_df["churn_probability"] >= 0.4) & (results_df["churn_probability"] < 0.7)).sum())
    low_risk = int((results_df["churn_probability"] < 0.4).sum())
    seg_counts = results_df["segment"].value_counts().to_dict()

    return {
        "success": True,
        "total_customers": len(results_df),
        "high_risk_count": high_risk,
        "moderate_risk_count": mod_risk,
        "low_risk_count": low_risk,
        "segments": {int(k): int(v) for k, v in seg_counts.items()},
        "issues": all_issues,
        "elapsed_seconds": round(elapsed, 2),
    }


@app.get("/status")
def api_pipeline_status():
    """Get ML pipeline last run status."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT predicted_at, COUNT(*) as prediction_count
            FROM customer_predictions
            GROUP BY predicted_at
            ORDER BY predicted_at DESC
            LIMIT 1
        """).fetchone()
        if row:
            return {
                "last_run": row["predicted_at"],
                "total_predictions": row["prediction_count"],
            }
    return {"last_run": None, "total_predictions": 0}


@app.get("/health")
def api_health():
    return {"status": "ok", "service": "ml_pipeline_api", "port": 8001}
