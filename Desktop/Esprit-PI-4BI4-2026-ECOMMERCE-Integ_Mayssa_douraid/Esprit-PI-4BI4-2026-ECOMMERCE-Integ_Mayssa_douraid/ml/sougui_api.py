from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from observability import set_model_metric
from observability import set_data_metric
from observability import set_drift_metric

router = APIRouter(tags=["sougui"])

SOUGUI_MODEL_PATH = Path(__file__).resolve().parent / "sougui_mlops" / "models" / "best_model.pkl"
SOUGUI_OUTPUTS_DIR = Path(__file__).resolve().parent / "sougui_mlops" / "outputs"
SOUGUI_EXTERNAL_OUTPUTS_DIR = Path.home() / "Downloads" / "Sougui_outputs"
_sougui_model: Any | None = None
_sougui_model_name: str = "unknown"


class SouguiPredictRequest(BaseModel):
    product: str
    avg_price: float
    feat_cat_popularity: float = 0.5
    feat_price_distance: float = 0.0
    feat_occurrence_score: float = 1.0
    feat_concurrent_coverage: float = 1.0


class SouguiPredictResponse(BaseModel):
    product: str
    predicted_category: str
    confidence: float
    model: str


def _sanitize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        if isinstance(value, float) and pd.isna(value):
            return None
        return value
    if pd.isna(value):
        return None
    return str(value)


def _sanitize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in records:
        cleaned.append({key: _sanitize_scalar(value) for key, value in row.items()})
    return cleaned


def _resolve_output_file(candidates: list[str]) -> Path | None:
    for base in [SOUGUI_OUTPUTS_DIR, SOUGUI_EXTERNAL_OUTPUTS_DIR]:
        for name in candidates:
            path = base / name
            if path.exists() and path.is_file():
                return path
    return None


def _read_csv_rows(path: Path, required_columns: list[str]) -> list[dict[str, Any]]:
    data = pd.read_csv(path)
    for column in required_columns:
        if column not in data.columns:
            data[column] = None
    return _sanitize_records(data[required_columns].to_dict(orient="records"))


def _ensure_sougui_model() -> tuple[Any, str]:
    global _sougui_model
    global _sougui_model_name

    if _sougui_model is not None:
        return _sougui_model, _sougui_model_name

    if not SOUGUI_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Sougui model file not found: {SOUGUI_MODEL_PATH}")

    try:
        package = joblib.load(SOUGUI_MODEL_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load Sougui model: {exc}") from exc

    model = package.get("model") if isinstance(package, dict) else None
    if model is None:
        raise HTTPException(status_code=500, detail="Sougui model package is invalid: key 'model' is missing.")

    _sougui_model = model
    _sougui_model_name = str(package.get("model_name", "sougui_model")) if isinstance(package, dict) else "sougui_model"
    return _sougui_model, _sougui_model_name


@router.get("/health/sougui")
def sougui_health() -> dict[str, Any]:
    if not SOUGUI_MODEL_PATH.exists():
        return {
            "status": "missing",
            "model_loaded": False,
            "model_path": str(SOUGUI_MODEL_PATH),
        }

    model_loaded = _sougui_model is not None
    model_name = _sougui_model_name if model_loaded else "sougui_model"
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model": model_name,
        "model_path": str(SOUGUI_MODEL_PATH),
        "outputs_dir": str(SOUGUI_OUTPUTS_DIR),
        "external_outputs_dir": str(SOUGUI_EXTERNAL_OUTPUTS_DIR),
    }


@router.post("/predict/sougui", response_model=SouguiPredictResponse)
def predict_sougui(payload: SouguiPredictRequest) -> SouguiPredictResponse:
    model, model_name = _ensure_sougui_model()

    df = pd.DataFrame(
        [
            {
                "product": payload.product.lower().strip(),
                "avg_price": payload.avg_price,
                "feat_cat_popularity": payload.feat_cat_popularity,
                "feat_price_distance": payload.feat_price_distance,
                "feat_occurrence_score": payload.feat_occurrence_score,
                "feat_concurrent_coverage": payload.feat_concurrent_coverage,
            }
        ]
    )
    try:
        prediction = str(model.predict(df)[0])
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(df).max())
        else:
            confidence = 0.0
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sougui prediction failed: {exc}") from exc

    set_model_metric("sougui", "confidence", confidence)
    set_drift_metric("sougui", "dummy_drift", 0.1)
    missing_ratio = df.isna().mean().mean()
    set_data_metric("sougui", "missing_ratio", missing_ratio)

    return {
        "product": payload.product,
        "predicted_category": prediction,
        "confidence": confidence,
        "model": model_name,
    }


@router.get("/predict/sougui/products")
def sougui_products_predictions() -> dict[str, Any]:
    columns = [
        "product",
        "category",
        "predicted_category",
        "confidence",
        "avg_price",
        "nb_concurrents",
        "product_score",
        "niveau_recommandation",
    ]

    csv_file = _resolve_output_file([
        "products_with_predictions.csv",
        "recommendations_produits_sougui_complet.csv",
    ])
    if csv_file is None:
        return {"count": 0, "predictions": []}

    rows = _read_csv_rows(csv_file, columns)
    for row in rows:
        row["suggested_action"] = row.get("niveau_recommandation")
    return {"count": len(rows), "predictions": rows}


@router.get("/top-products/sougui")
def sougui_top_products() -> dict[str, Any]:
    columns = [
        "product",
        "category",
        "avg_price",
        "nb_concurrents",
        "confidence_score",
        "product_score",
        "niveau_recommandation",
    ]

    csv_file = _resolve_output_file([
        "recommendations_produits_sougui_top100.csv",
        "recommendations_produits_sougui_complet.csv",
        "products_with_predictions.csv",
    ])
    if csv_file is None:
        return {"data": []}

    rows = _read_csv_rows(csv_file, columns)
    rows = rows[:50]
    return {"data": rows}


@router.get("/categories/sougui")
def sougui_categories() -> dict[str, Any]:
    columns = [
        "category",
        "nb_produits_sougui",
        "nb_produits_conc",
        "coverage_ratio",
        "positionnement_couverture",
        "avg_price_conc",
        "opportunity_score",
        "niveau_opportunite",
    ]

    csv_file = _resolve_output_file([
        "recommendations_categories_sougui.csv",
    ])
    if csv_file is None:
        return {"data": []}

    rows = _read_csv_rows(csv_file, columns)
    return {"data": rows}