"""Churn prediction FastAPI application.

Provides prediction endpoints, training pipeline, PCA visualization,
and model status. Runs on port 8001 as a separate service.

Data sources: PostgreSQL (primary) → CSV files (fallback).
"""
from __future__ import annotations

import logging
import math
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from prometheus_client import Counter
from pydantic import BaseModel

from api_common import safe_head
from churn_pipeline import (
    ChurnPipeline,
    SEGMENT_LABELS,
    SEGMENT_PROFILES,
    DEFAULT_ARTIFACTS_DIR,
    get_pipeline,
)
from observability import (
    log_event,
    publish_churn_metrics,
    publish_data_metrics,
    publish_drift_metrics,
    publish_model_metrics,
    record_churn_prediction,
    record_error,
    record_retraining_trigger,
    setup_metrics,
)
from train_pipeline import train_churn_pipeline

CHURN_MODEL_NAME = "churn_prediction"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Churn Prediction API",
    description="Customer churn prediction + segmentation service",
    version="1.0.0",
)

setup_metrics(app)

CHURN_ERROR_COUNT = Counter("churn_http_errors_total", "Total number of churn API errors")

router = APIRouter(prefix="/predict", tags=["churn"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    recency: float
    frequency: float
    monetary_total: float
    monetary_trend: float
    product_diversity: float
    channel_diversity: float
    lifetime_days: float
    purchase_velocity: float
    avg_price: float
    channel: str = ""


class CustomerPrediction(BaseModel):
    churn_probability: float
    churn_prediction: str
    segment: int
    segment_label: str
    risk_level: str
    recommendation: str


class BatchPredictionResponse(BaseModel):
    predictions: list[CustomerPrediction]
    count: int
    avg_probability: float
    high_risk_count: int
    segment_distribution: dict[str, int]
    issues: list[str]


class TrainResponse(BaseModel):
    run_uuid: str
    data_source: str
    n_customers: int
    n_transactions: int
    churn_model: str
    churn_metrics: dict[str, Any]
    segment_model: str
    segment_metrics: dict[str, Any]
    mlflow_available: bool
    saved_paths: list[str]
    derivation_issues: list[str]


class TrainStatusResponse(BaseModel):
    ready: bool
    artifacts_dir: str
    models_loaded: bool
    known_segments: int
    known_channels: list[str]


class PCAPlotResponse(BaseModel):
    available: bool
    note: str = ""


class ModelStatusResponse(BaseModel):
    artifacts_dir: str
    churn_model: bool
    churn_scaler: bool
    segment_model: bool
    segment_scaler: bool
    config: bool
    reference_date: str = ""
    decision_threshold: float = 0.0


class TransactionFeatures(BaseModel):
    customerFK: str = ""
    FactVentePK: int = 1
    dateFK: str | int = ""
    total_ttc: float = 0.0
    quantity: int = 1
    price: float = 0.0
    channelFK: str | int = ""
    productFK: str | int = ""


class ChurnDriftResponse(BaseModel):
    drift_detected: bool
    avg_probability_shift: float
    baseline_avg_probability: float | None = None
    current_avg_probability: float | None = None
    high_risk_rate: float = 0.0
    baseline_high_risk_rate: float | None = None
    note: str | None = None


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def count_churn_errors(request, call_next):
    try:
        response = await call_next(request)
        if response.status_code >= 400:
            CHURN_ERROR_COUNT.inc()
        return response
    except Exception:
        CHURN_ERROR_COUNT.inc()
        raise


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=CustomerPrediction)
def predict_churn(features: CustomerFeatures) -> CustomerPrediction:
    """Predict churn probability and segment for a single customer."""
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    features_dict = features.model_dump() if hasattr(features, "model_dump") else features.dict()
    result = pipeline.predict_customer(features_dict)
    prediction = CustomerPrediction(**result)

    record_churn_prediction(
        CHURN_MODEL_NAME,
        churn_probability=prediction.churn_probability,
        segment=prediction.segment,
        risk_level=prediction.risk_level,
    )
    publish_churn_metrics(
        CHURN_MODEL_NAME,
        avg_probability=prediction.churn_probability,
        high_risk_rate=1.0 if prediction.churn_probability >= 0.7 else 0.0,
    )

    return prediction


@router.post("/batch", response_model=BatchPredictionResponse)
def predict_churn_batch(features_list: list[CustomerFeatures]) -> BatchPredictionResponse:
    """Batch predict churn for multiple customers."""
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    features_dicts = [
        f.model_dump() if hasattr(f, "model_dump") else f.dict()
        for f in features_list
    ]
    df = pipeline._coerce_numeric_frame(
        pipeline._ensure_log_features(
            pipeline._encode_channel_dummies(
                pd.DataFrame(features_dicts),
                features_list[0].channel if features_list else ""
            )
        ),
        pipeline.config["churn_feature_columns"]
    )[0]

    results_df, issues = pipeline.batch_predict(df, features_list[0].channel if features_list else "")

    predictions = []
    for _, row in results_df.iterrows():
        predictions.append(CustomerPrediction(
            churn_probability=float(row["churn_probability"]),
            churn_prediction=str(row["churn_prediction"]),
            segment=int(row["segment"]),
            segment_label=str(row["segment_label"]),
            risk_level=str(row["risk_level"]),
            recommendation=str(row["recommendation"]),
        ))

    proba_series = results_df["churn_probability"]
    segment_dist = results_df["segment"].value_counts().to_dict()
    segment_dist_str = {str(k): int(v) for k, v in segment_dist.items()}

    avg_prob = float(proba_series.mean()) if len(proba_series) > 0 else 0.0
    high_risk_cnt = int((proba_series >= 0.7).sum())
    high_risk_rate = high_risk_cnt / len(predictions) if predictions else 0.0

    publish_churn_metrics(
        CHURN_MODEL_NAME,
        avg_probability=avg_prob,
        high_risk_rate=high_risk_rate,
        prediction_count=len(predictions),
    )

    log_event(
        logging.INFO,
        "churn_batch_prediction",
        model=CHURN_MODEL_NAME,
        count=len(predictions),
        avg_probability=avg_prob,
        high_risk_count=high_risk_cnt,
        high_risk_rate=high_risk_rate,
    )

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        avg_probability=avg_prob,
        high_risk_count=high_risk_cnt,
        segment_distribution=segment_dist_str,
        issues=issues,
    )


@router.post("/batch/csv")
async def predict_churn_batch_csv(file: UploadFile) -> dict[str, Any]:
    """Batch predict from uploaded CSV with prepared features."""
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    channel_value = ""
    if "channel" in df.columns:
        channel_value = str(df["channel"].iloc[0]) if df["channel"].notna().any() else ""

    results_df, issues = pipeline.batch_predict(df, channel_value)

    records = []
    for _, row in results_df.iterrows():
        records.append({
            "customerFK": str(row.get("customerFK", "")),
            "churn_probability": float(row["churn_probability"]),
            "churn_prediction": str(row["churn_prediction"]),
            "segment": int(row["segment"]),
            "segment_label": str(row["segment_label"]),
            "risk_level": str(row["risk_level"]),
            "recommendation": str(row["recommendation"]),
        })

    proba_series = results_df["churn_probability"]

    return {
        "count": len(records),
        "avg_probability": float(proba_series.mean()) if len(proba_series) > 0 else 0.0,
        "high_risk_count": int((proba_series >= 0.7).sum()),
        "segment_distribution": results_df["segment"].value_counts().to_dict(),
        "issues": issues,
        "predictions": records,
    }


@router.post("/batch/transactions")
async def predict_from_transactions(file: UploadFile) -> dict[str, Any]:
    """Derive features from raw transactions and predict."""
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    try:
        customer_df, issues = pipeline.derive_features_from_transactions(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature derivation failed: {e}")

    if customer_df.empty:
        raise HTTPException(status_code=400, detail="No valid customers after feature derivation")

    results_df, inference_issues = pipeline.batch_predict(customer_df, "")
    all_issues = issues + inference_issues

    records = []
    for _, row in results_df.iterrows():
        records.append({
            "customerFK": str(row.get("customerFK", "")),
            "recency": float(row.get("recency", 0)),
            "frequency": float(row.get("frequency", 0)),
            "monetary_total": float(row.get("monetary_total", 0)),
            "churn_probability": float(row["churn_probability"]),
            "churn_prediction": str(row["churn_prediction"]),
            "segment": int(row["segment"]),
            "segment_label": str(row["segment_label"]),
            "risk_level": str(row["risk_level"]),
            "recommendation": str(row["recommendation"]),
        })

    proba_series = results_df["churn_probability"]

    return {
        "n_customers": len(records),
        "n_transactions": len(df),
        "avg_probability": float(proba_series.mean()) if len(proba_series) > 0 else 0.0,
        "high_risk_count": int((proba_series >= 0.7).sum()),
        "issues": all_issues,
        "predictions": records,
    }


@router.post("/train", response_model=TrainResponse)
def train_churn(
    algorithm: str = "logistic_regression",
    n_clusters: int = 4,
    random_state: int = 42,
    decision_threshold: float = 0.5,
) -> TrainResponse:
    """Run the full training pipeline with MLflow tracking."""
    record_retraining_trigger(CHURN_MODEL_NAME, "manual_train_endpoint", algorithm=algorithm, n_clusters=n_clusters)

    try:
        result = train_churn_pipeline(
            algorithm=algorithm,
            n_clusters=n_clusters,
            random_state=random_state,
            decision_threshold=decision_threshold,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    get_pipeline.cache_clear()

    log_event(
        logging.INFO,
        "churn_training_completed",
        model=CHURN_MODEL_NAME,
        algorithm=algorithm,
        n_clusters=n_clusters,
        n_customers=result.get("n_customers", 0),
    )

    return TrainResponse(**result)


@router.get("/pca-plot")
def get_pca_plot() -> Any:
    """Return PCA segmentation plot as PNG image."""
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    plot_bytes = pipeline.generate_pca_plot()
    if plot_bytes is None:
        raise HTTPException(
            status_code=404,
            detail="PCA plot not available. Ensure customer_churn_predictions.csv exists."
        )

    return Response(content=plot_bytes, media_type="image/png")


@router.get("/segment-labels")
def get_segment_labels() -> dict[str, Any]:
    """Return segment labels and profiles."""
    return {
        "segments": {
            str(k): {"label": v, "profiles": SEGMENT_PROFILES.get(k, {})}
            for k, v in SEGMENT_LABELS.items()
        },
        "count": len(SEGMENT_LABELS),
    }


@router.get("/models", response_model=ModelStatusResponse)
def model_status() -> ModelStatusResponse:
    """Check model artifact status."""
    artifacts_dir = DEFAULT_ARTIFACTS_DIR
    base = Path(artifacts_dir)

    try:
        pipeline = get_pipeline()
        models_loaded = True
        reference_date = str(pipeline.reference_date)
        decision_threshold = pipeline.decision_threshold
        known_channels = pipeline.known_channels
    except FileNotFoundError:
        models_loaded = False
        reference_date = ""
        decision_threshold = 0.0
        known_channels = []

    return ModelStatusResponse(
        artifacts_dir=artifacts_dir,
        churn_model=(base / "churn_model.joblib").exists(),
        churn_scaler=(base / "churn_scaler.joblib").exists(),
        segment_model=(base / "segment_model.joblib").exists(),
        segment_scaler=(base / "segment_scaler.joblib").exists(),
        config=(base / "inference_config.json").exists(),
        reference_date=reference_date,
        decision_threshold=decision_threshold,
    )


@router.get("/status", response_model=TrainStatusResponse)
def status() -> TrainStatusResponse:
    """Check if the churn prediction service is ready."""
    try:
        pipeline = get_pipeline()
        return TrainStatusResponse(
            ready=True,
            artifacts_dir=DEFAULT_ARTIFACTS_DIR,
            models_loaded=True,
            known_segments=len(SEGMENT_LABELS),
            known_channels=pipeline.known_channels,
        )
    except FileNotFoundError:
        return TrainStatusResponse(
            ready=False,
            artifacts_dir=DEFAULT_ARTIFACTS_DIR,
            models_loaded=False,
            known_segments=0,
            known_channels=[],
        )


# ---------------------------------------------------------------------------
# Register router on app
# ---------------------------------------------------------------------------

app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "churn-prediction"}


@app.get("/train/status")
def train_status() -> dict[str, Any]:
    """Training pipeline status (separate from inference status)."""
    artifacts_dir = DEFAULT_ARTIFACTS_DIR
    base = Path(artifacts_dir)
    return {
        "artifacts_dir": artifacts_dir,
        "ready": (
            base.joinpath("churn_model.joblib").exists()
            and base.joinpath("segment_model.joblib").exists()
        ),
        "models": {
            "churn_model": base.joinpath("churn_model.joblib").exists(),
            "churn_scaler": base.joinpath("churn_scaler.joblib").exists(),
            "segment_model": base.joinpath("segment_model.joblib").exists(),
            "segment_scaler": base.joinpath("segment_scaler.joblib").exists(),
            "config": base.joinpath("inference_config.json").exists(),
        },
    }


@app.get("/drift/churn", response_model=ChurnDriftResponse)
def detect_churn_drift() -> ChurnDriftResponse:
    """Detect drift in churn prediction distribution.

    Compares current prediction distribution against a sliding window baseline.
    Drift is flagged if the average churn probability shifts by more than 5%.
    """
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    predictions_csv = Path("customer_churn_predictions.csv")
    if not predictions_csv.exists():
        publish_drift_metrics(
            CHURN_MODEL_NAME,
            {"accuracy_drop_ratio": 0.0, "detected": 0.0},
        )
        return ChurnDriftResponse(
            drift_detected=False,
            avg_probability_shift=0.0,
            note="No historical predictions found. Baseline not yet established.",
        )

    try:
        hist_df = pd.read_csv(predictions_csv)
    except Exception:
        hist_df = pd.DataFrame()

    if hist_df.empty or "churn_probability" not in hist_df.columns:
        publish_drift_metrics(
            CHURN_MODEL_NAME,
            {"accuracy_drop_ratio": 0.0, "detected": 0.0},
        )
        return ChurnDriftResponse(
            drift_detected=False,
            avg_probability_shift=0.0,
            note="No valid churn probability data in historical predictions.",
        )

    proba_col = pd.to_numeric(hist_df["churn_probability"], errors="coerce")
    total = len(proba_col.dropna())
    if total == 0:
        publish_drift_metrics(
            CHURN_MODEL_NAME,
            {"accuracy_drop_ratio": 0.0, "detected": 0.0},
        )
        return ChurnDriftResponse(
            drift_detected=False,
            avg_probability_shift=0.0,
            note="No valid probability values.",
        )

    recent_n = max(50, total // 4)
    sorted_df = hist_df.sort_index()
    baseline_probas = pd.to_numeric(sorted_df["churn_probability"].iloc[:total - recent_n], errors="coerce").dropna()
    current_probas = pd.to_numeric(sorted_df["churn_probability"].iloc[total - recent_n:], errors="coerce").dropna()

    if len(baseline_probas) == 0 or len(current_probas) == 0:
        publish_drift_metrics(
            CHURN_MODEL_NAME,
            {"accuracy_drop_ratio": 0.0, "detected": 0.0},
        )
        return ChurnDriftResponse(
            drift_detected=False,
            avg_probability_shift=0.0,
            note="Insufficient data for baseline vs current comparison.",
        )

    baseline_avg = float(baseline_probas.mean())
    current_avg = float(current_probas.mean())
    baseline_high_risk = float((baseline_probas >= 0.7).mean())
    current_high_risk = float((current_probas >= 0.7).mean())

    avg_shift = abs(current_avg - baseline_avg)
    reference = max(baseline_avg, 1e-9)
    accuracy_drop_ratio = avg_shift / reference

    drift_detected = accuracy_drop_ratio > 0.05

    publish_model_metrics(
        CHURN_MODEL_NAME,
        {
            "avg_churn_probability": current_avg,
            "high_risk_rate": current_high_risk,
        },
    )
    publish_drift_metrics(
        CHURN_MODEL_NAME,
        {
            "accuracy_drop_ratio": accuracy_drop_ratio,
            "detected": 1.0 if drift_detected else 0.0,
        },
    )

    log_event(
        logging.WARNING if drift_detected else logging.INFO,
        "churn_drift_check",
        drift_detected=drift_detected,
        avg_probability_shift=avg_shift,
        accuracy_drop_ratio=accuracy_drop_ratio,
        baseline_avg_probability=baseline_avg,
        current_avg_probability=current_avg,
        baseline_high_risk_rate=baseline_high_risk,
        current_high_risk_rate=current_high_risk,
    )

    if drift_detected:
        log_event(
            logging.WARNING,
            "churn_drift_detected",
            reason="probability_distribution_shift",
            avg_probability_shift=avg_shift,
            accuracy_drop_ratio=accuracy_drop_ratio,
        )

    return ChurnDriftResponse(
        drift_detected=drift_detected,
        avg_probability_shift=round(avg_shift, 4),
        baseline_avg_probability=round(baseline_avg, 4) if not math.isnan(baseline_avg) else None,
        current_avg_probability=round(current_avg, 4) if not math.isnan(current_avg) else None,
        high_risk_rate=round(current_high_risk, 4),
        baseline_high_risk_rate=round(baseline_high_risk, 4) if not math.isnan(baseline_high_risk) else None,
        note="Churn probability distribution shifted." if drift_detected else "No drift detected.",
    )
