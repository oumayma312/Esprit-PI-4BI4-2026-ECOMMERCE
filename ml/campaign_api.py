import json
import math
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from campaign_business_rules import evaluate_campaign_prediction
from campaign_success_predictionn import SUMMARY_PATH, run_pipeline

router = APIRouter(tags=["campaign"])

CAMPAIGN_SUMMARY_PATH = Path(SUMMARY_PATH)
_campaign_latest_summary: dict[str, Any] | None = None


class CampaignPrepareDatasetResponse(BaseModel):
    rows: int
    feature_columns: list[str]
    preview: list[dict[str, Any]]


class CampaignTrainResponse(BaseModel):
    summary_path: str
    rows: int
    target_rule: str | None = None
    classification_best_model: str | None = None
    regression_best_model: str | None = None
    clustering_best_k: int | None = None
    mlflow: dict[str, Any] | None = None


class CampaignTrainResultsResponse(BaseModel):
    available: bool
    results: dict[str, Any] | None = None


class CampaignPredictResponse(BaseModel):
    available: bool
    summary_path: str
    summary: dict[str, Any] | None = None
    note: str | None = None


class CampaignFormPredictRequest(BaseModel):
    reach: float = 0.0
    impressions: float = 0.0
    frequency: float = 0.0
    result: float = 0.0
    views: float = 0.0
    price: float = 0.0


class CampaignFormPredictResponse(BaseModel):
    predicted_success: int
    probability: float
    decision: str
    strategy: str
    model_reference: str | None = None
    score: float
    highlights: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    derived_metrics: dict[str, float | None] = Field(default_factory=dict)
    inputs: dict[str, float]


def _json_safe(value: Any) -> Any:
    # Convert numpy/pandas scalars when possible.
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    return str(value)


def _summary_from_disk() -> dict[str, Any] | None:
    if not CAMPAIGN_SUMMARY_PATH.exists():
        return None
    try:
        parsed = json.loads(CAMPAIGN_SUMMARY_PATH.read_text(encoding="utf-8"))
        return _json_safe(parsed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Campaign summary read failed: {exc}") from exc


def _get_cached_or_disk_summary() -> dict[str, Any] | None:
    global _campaign_latest_summary
    if isinstance(_campaign_latest_summary, dict):
        return _campaign_latest_summary
    disk_summary = _summary_from_disk()
    if isinstance(disk_summary, dict):
        _campaign_latest_summary = disk_summary
        return disk_summary
    return None


def _execute_campaign_pipeline() -> dict[str, Any]:
    global _campaign_latest_summary
    try:
        summary = run_pipeline()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Campaign pipeline execution failed: {exc}") from exc

    if not isinstance(summary, dict):
        raise HTTPException(status_code=500, detail="Campaign pipeline returned an invalid payload.")

    safe_summary = _json_safe(summary)
    _campaign_latest_summary = safe_summary
    return safe_summary


@router.get("/train/campaign/dataset", response_model=CampaignPrepareDatasetResponse)
def get_campaign_dataset(limit: int = 20) -> CampaignPrepareDatasetResponse:
    n = max(1, min(limit, 5000))
    summary = _get_cached_or_disk_summary()
    if not isinstance(summary, dict):
        summary = _execute_campaign_pipeline()

    dataset = summary.get("dataset") if isinstance(summary, dict) else {}
    rows = int(dataset.get("rows", 0)) if isinstance(dataset, dict) else 0
    columns = dataset.get("columns", []) if isinstance(dataset, dict) else []
    preview = dataset.get("head", []) if isinstance(dataset, dict) else []

    if rows <= 0:
        raise HTTPException(status_code=404, detail="No campaign dataset available. Run training first.")

    return {
        "rows": rows,
        "feature_columns": [str(c) for c in columns[:n]],
        "preview": preview[:n],
    }


@router.post("/train/campaign/train", response_model=CampaignTrainResponse)
def train_campaign_model() -> CampaignTrainResponse:
    summary = _execute_campaign_pipeline()
    dataset = summary.get("dataset", {}) if isinstance(summary, dict) else {}
    classification = summary.get("classification") if isinstance(summary.get("classification"), dict) else {}
    regression = summary.get("regression") if isinstance(summary.get("regression"), dict) else {}
    clustering = summary.get("clustering") if isinstance(summary.get("clustering"), dict) else {}

    return {
        "summary_path": str(CAMPAIGN_SUMMARY_PATH),
        "rows": int(dataset.get("rows", 0)),
        "target_rule": summary.get("target_rule"),
        "classification_best_model": classification.get("best_model"),
        "regression_best_model": regression.get("best_model"),
        "clustering_best_k": clustering.get("best_k"),
        "mlflow": summary.get("mlflow") if isinstance(summary.get("mlflow"), dict) else None,
    }


@router.get("/train/campaign/results", response_model=CampaignTrainResultsResponse)
def get_campaign_train_results() -> CampaignTrainResultsResponse:
    summary = _get_cached_or_disk_summary()
    return {"available": isinstance(summary, dict), "results": summary}


@router.get("/predict/campaign", response_model=CampaignPredictResponse)
def predict_campaign() -> CampaignPredictResponse:
    summary = _get_cached_or_disk_summary()
    if not isinstance(summary, dict):
        summary = _execute_campaign_pipeline()

    return {
        "available": isinstance(summary, dict),
        "summary_path": str(CAMPAIGN_SUMMARY_PATH),
        "summary": summary,
        "note": "Campaign prediction summary generated from the MLOps pipeline.",
    }


@router.post("/predict/campaign/form", response_model=CampaignFormPredictResponse)
def predict_campaign_from_form(payload: CampaignFormPredictRequest) -> CampaignFormPredictResponse:
    summary = _get_cached_or_disk_summary()
    if not isinstance(summary, dict):
        summary = _execute_campaign_pipeline()

    model_reference = None
    if isinstance(summary.get("classification"), dict):
        model_reference = summary["classification"].get("best_model")
    if not model_reference and isinstance(summary.get("regression"), dict):
        model_reference = summary["regression"].get("best_model")

    try:
        evaluation = evaluate_campaign_prediction(
            payload.model_dump(),
            summary.get("business_context") if isinstance(summary, dict) else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "predicted_success": evaluation["predicted_success"],
        "probability": evaluation["probability"],
        "decision": evaluation["decision"],
        "strategy": evaluation["strategy"],
        "model_reference": model_reference,
        "score": evaluation["score"],
        "highlights": evaluation["highlights"],
        "warnings": evaluation["warnings"],
        "derived_metrics": evaluation["derived_metrics"],
        "inputs": evaluation["inputs"],
    }
