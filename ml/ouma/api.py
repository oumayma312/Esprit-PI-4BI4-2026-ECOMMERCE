from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(prefix="/ouma", tags=["Ouma Product Performance"])

base_dir = Path(__file__).resolve().parent

model_path = base_dir / "models" / "product_model.pkl"
columns_path = base_dir / "artifacts" / "training_columns.pkl"

model = joblib.load(model_path)
training_columns = joblib.load(columns_path)


class ProductPredictionInput(BaseModel):
    total_quantity: float = 0
    avg_price: float = 0
    revenue: float = 0
    month_num: int = 1
    categoryFK: str = "-1"
    order_count: float = 0


@router.get("/health")
def ouma_health():
    return {
        "status": "ok",
        "module": "ouma_product_performance"
    }


@router.post("/predict/one")
def predict_one(data: ProductPredictionInput):
    input_df = pd.DataFrame([{
        "total_quantity": data.total_quantity,
        "avg_price": data.avg_price,
        "revenue": data.revenue,
        "month_num": data.month_num,
        "categoryFK": str(data.categoryFK),
        "order_count": data.order_count
    }])

    input_df = pd.get_dummies(input_df, columns=["categoryFK"], drop_first=True)
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    predicted_class = model.predict(input_df)[0]
    suggested_action = "Promote" if predicted_class == "High" else "Monitor"

    return {
        "predicted_next_month_class": predicted_class,
        "suggested_action": suggested_action
    }