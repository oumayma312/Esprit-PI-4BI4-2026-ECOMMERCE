from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from observability import ml_model_metric_value

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_model.pkl"

app = FastAPI(title="Sougui ML API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/")
def home():
    return {
        "message": "Sougui ML API is running",
        "model": model_name
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model": model_name
    }


@app.post("/predict")
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

    ml_model_metric_value.labels(
    model="sougui",
    metric="confidence"
).set(confidence)

    return {
        "product": input_data.product,
        "predicted_category": prediction,
        "confidence": confidence,
        "model": model_name
    }
