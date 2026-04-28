"""Churn prediction FastAPI application.

Provides prediction endpoints, training pipeline, PCA visualization,
and model status. Runs on port 8001 as a separate service.

Data sources: PostgreSQL (primary) → CSV files (fallback).
"""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from api_common import safe_head
from churn_pipeline import (
    ChurnPipeline,
    SEGMENT_LABELS,
    SEGMENT_PROFILES,
    DEFAULT_ARTIFACTS_DIR,
    get_pipeline,
)
from train_pipeline import train_churn_pipeline


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Churn Prediction API",
    description="Customer churn prediction + segmentation service",
    version="1.0.0",
)

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

    return CustomerPrediction(**result)


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

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        avg_probability=float(proba_series.mean()) if len(proba_series) > 0 else 0.0,
        high_risk_count=int((proba_series >= 0.7).sum()),
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
