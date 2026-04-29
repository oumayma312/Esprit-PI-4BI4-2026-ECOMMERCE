import joblib
import pandas as pd
from pathlib import Path
from fastapi import APIRouter
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

router = APIRouter(prefix="/sougui", tags=["Sougui ML"])

model_package = joblib.load(MODEL_PATH)
model = model_package["model"]
model_name = model_package["model_name"]
numeric_features = model_package["numeric_features"]


class ProductInput(BaseModel):
    product: str
    avg_price: float
    feat_cat_popularity: float = 0.5
    feat_price_distance: float = 0.0
    feat_occurrence_score: float = 1.0
    feat_concurrent_coverage: float = 1.0


@router.get("/")
def home():
    return {
        "message": "Sougui ML module is running",
        "model": model_name
    }


@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model": model_name
    }


@router.post("/predict")
def predict(input_data: ProductInput):
    df = pd.DataFrame([{
        "product": input_data.product.lower().strip(),
        "avg_price": input_data.avg_price,
        "feat_cat_popularity": input_data.feat_cat_popularity,
        "feat_price_distance": input_data.feat_price_distance,
        "feat_occurrence_score": input_data.feat_occurrence_score,
        "feat_concurrent_coverage": input_data.feat_concurrent_coverage,
    }])

    prediction = model.predict(df)[0]
    confidence = float(model.predict_proba(df).max())

    return {
        "product": input_data.product,
        "predicted_category": prediction,
        "confidence": confidence,
        "model": model_name
    }