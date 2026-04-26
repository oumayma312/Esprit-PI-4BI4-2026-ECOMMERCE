import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine, text

from api_common import (
    SELL_MODEL_PATH,
    compute_sell_recommendation_from_forecast,
    forecast_from_artifact_model,
    load_artifact,
    load_dataset_from_sources,
    save_versioned_artifact,
    safe_head,
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

try:
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    MLFLOW_AVAILABLE = False

router = APIRouter(tags=["sell"])

SELL_TRAIN_MODEL_PATH = "sell_training_model_api.pkl"
SELL_NOTEBOOK_DATASET_PATH = "sell_feature_dataset.parquet"
SELL_DB_DATASET_TABLE = "sell"
_sell_prepared_dataset: pd.DataFrame | None = None
_sell_train_results: dict[str, Any] | None = None


class SellRecommendResponse(BaseModel):
    best_model: str | None = None
    best_month: str | None = None
    best_date: str | None = None
    best_value: float | None = None
    note: str | None = None


class ForecastSellResponse(BaseModel):
    best_model: str | None = None
    rows: list[dict[str, Any]]
    note: str | None = None


class SellPrepareDatasetResponse(BaseModel):
    rows: int
    feature_columns: list[str]
    preview: list[dict[str, Any]]


class SellTrainResponse(BaseModel):
    model_path: str
    versioned_model_path: str | None = None
    rows_train: int
    rows_test: int
    metrics: dict[str, float]


class SellTrainResultsResponse(BaseModel):
    available: bool
    results: dict[str, Any] | None = None


class SellPredictResponse(BaseModel):
    best_model: str | None = None
    best_month: str | None = None
    best_date: str | None = None
    best_value: float | None = None
    rows: list[dict[str, Any]]
    note: str | None = None


class SellMonthlySaveResponse(BaseModel):
    saved_rows: int
    run_id: str
    target_table: str


def _prepare_sell_features(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns and "ds" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing date column. Provide 'date' or 'ds'.")
    if "revenue" not in df.columns and "y" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing revenue column. Provide 'revenue' or 'y'.")

    work = df.copy()
    date_col = "date" if "date" in work.columns else "ds"
    y_col = "revenue" if "revenue" in work.columns else "y"

    work["ds"] = pd.to_datetime(work[date_col], errors="coerce")
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=["ds", "y"]).copy()
    if work.empty:
        raise HTTPException(status_code=400, detail="No valid rows after parsing date/revenue.")

    daily = work.groupby("ds", as_index=False)["y"].sum().sort_values("ds")
    daily["dow"] = daily["ds"].dt.weekday
    daily["month"] = daily["ds"].dt.month
    daily["lag_1"] = daily["y"].shift(1)
    daily["lag_7"] = daily["y"].shift(7)
    daily["lag_14"] = daily["y"].shift(14)
    daily["roll_mean_7"] = daily["y"].shift(1).rolling(7).mean()
    daily = daily.dropna().reset_index(drop=True)

    if daily.empty:
        raise HTTPException(status_code=400, detail="Not enough history to create lag features.")

    return daily


def _qident_local(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _build_pg_engine_from_env() -> Any:
    db_host = os.getenv("PGHOST") or os.getenv("DB_HOST")
    db_port = os.getenv("PGPORT") or os.getenv("DB_PORT") or "5432"
    db_name = os.getenv("PGDATABASE") or os.getenv("DB_NAME")
    db_user = os.getenv("PGUSER") or os.getenv("DB_USER")
    db_password = os.getenv("PGPASSWORD") or os.getenv("DB_PASSWORD")

    needed = [db_host, db_name, db_user, db_password]
    if any(v is None or str(v).strip() == "" for v in needed):
        missing = [
            name
            for name, value in [
                ("PGHOST/DB_HOST", db_host),
                ("PGDATABASE/DB_NAME", db_name),
                ("PGUSER/DB_USER", db_user),
                ("PGPASSWORD/DB_PASSWORD", db_password),
            ]
            if value is None or str(value).strip() == ""
        ]
        raise HTTPException(
            status_code=503,
            detail=f"PostgreSQL configuration is missing. Missing: {', '.join(missing)}",
        )

    return create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
        pool_pre_ping=True,
    )


def _log_sell_mlflow_run(
    *,
    run_name: str,
    model,
    feature_columns: list[str],
    random_state: int,
    rows_train: int,
    rows_test: int,
    metrics: dict[str, float],
    comparison_label: str,
    n_estimators: int,
) -> None:
    if not MLFLOW_AVAILABLE:
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("sell-forecasting")

    payload = {
        "task": "sell_train_api",
        "model": model,
        "feature_columns": feature_columns,
        "random_state": int(random_state),
        "n_estimators": int(n_estimators),
    }
    artifact_dir = Path("./mlartifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{run_name}.joblib"
    joblib.dump(payload, artifact_path)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "random_state": int(random_state),
                "n_estimators": int(n_estimators),
                "rows_train": int(rows_train),
                "rows_test": int(rows_test),
                "feature_count": int(len(feature_columns)),
                "comparison_label": comparison_label,
            }
        )
        mlflow.log_metrics(
            {
                "mae": float(metrics.get("mae", float("nan"))),
                "rmse": float(metrics.get("rmse", float("nan"))),
                "mape": float(metrics.get("mape", float("nan"))),
            }
        )
        mlflow.log_artifact(str(artifact_path), artifact_path="artifacts")


@router.get("/train/sell/dataset", response_model=SellPrepareDatasetResponse)
def get_sell_dataset(limit: int = 20) -> SellPrepareDatasetResponse:
    n = max(1, min(limit, 5000))
    ds = load_dataset_from_sources(_sell_prepared_dataset, SELL_DB_DATASET_TABLE, SELL_NOTEBOOK_DATASET_PATH)
    if ds is None or ds.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "No prepared sell dataset found in API memory, PostgreSQL, or notebook snapshot. "
                "Expected PostgreSQL table ml.sell or file sell_feature_dataset.parquet."
            ),
        )

    feature_cols = [c for c in ds.columns if c not in ["ds", "y"]]
    return {
        "rows": int(len(ds)),
        "feature_columns": feature_cols,
        "preview": safe_head(ds, n=n),
    }


@router.post("/train/sell/train", response_model=SellTrainResponse)
def train_sell_model(random_state: int = 42) -> SellTrainResponse:
    global _sell_prepared_dataset, _sell_train_results

    ds = load_dataset_from_sources(_sell_prepared_dataset, SELL_DB_DATASET_TABLE, SELL_NOTEBOOK_DATASET_PATH)
    if ds is None or ds.empty:
        raise HTTPException(
            status_code=404,
            detail="No sell dataset available. Check PostgreSQL table ml.sell or the snapshot file.",
        )

    _sell_prepared_dataset = ds

    ds = ds.copy().sort_values("ds")
    feature_cols = [c for c in ds.columns if c not in ["ds", "y"]]
    split_idx = max(1, int(len(ds) * 0.8))
    train_df = ds.iloc[:split_idx]
    test_df = ds.iloc[split_idx:]
    if test_df.empty:
        raise HTTPException(status_code=400, detail="Not enough rows to create train/test split.")

    X_train = train_df[feature_cols]
    y_train = train_df["y"]
    X_test = test_df[feature_cols]
    y_test = test_df["y"]

    selected_n_estimators = 300
    comparison_n_estimators = 150 if selected_n_estimators != 150 else 200

    model = RandomForestRegressor(n_estimators=selected_n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mask = np.abs(y_test.values) > 1e-9
    mape = float(np.mean(np.abs((y_test.values[mask] - pred[mask]) / y_test.values[mask])) * 100) if mask.any() else float("nan")

    versioned_model_path = save_versioned_artifact({"model": model, "feature_columns": feature_cols, "task": "sell_train_api"}, SELL_TRAIN_MODEL_PATH)

    selected_metrics = {"mae": mae, "rmse": rmse, "mape": mape}

    comparison_model = RandomForestRegressor(n_estimators=comparison_n_estimators, random_state=random_state)
    comparison_model.fit(X_train, y_train)
    comparison_pred = comparison_model.predict(X_test)
    comparison_mae = float(mean_absolute_error(y_test, comparison_pred))
    comparison_rmse = float(np.sqrt(mean_squared_error(y_test, comparison_pred)))
    comparison_mape = (
        float(np.mean(np.abs((y_test.values[mask] - comparison_pred[mask]) / y_test.values[mask])) * 100)
        if mask.any()
        else float("nan")
    )

    _log_sell_mlflow_run(
        run_name=f"sell_rf_n{selected_n_estimators}",
        model=model,
        feature_columns=feature_cols,
        random_state=random_state,
        rows_train=len(train_df),
        rows_test=len(test_df),
        metrics=selected_metrics,
        comparison_label="selected",
        n_estimators=selected_n_estimators,
    )
    _log_sell_mlflow_run(
        run_name=f"sell_rf_n{comparison_n_estimators}",
        model=comparison_model,
        feature_columns=feature_cols,
        random_state=random_state,
        rows_train=len(train_df),
        rows_test=len(test_df),
        metrics={"mae": comparison_mae, "rmse": comparison_rmse, "mape": comparison_mape},
        comparison_label="comparison",
        n_estimators=comparison_n_estimators,
    )

    _sell_train_results = {
        "model_path": SELL_TRAIN_MODEL_PATH,
        "versioned_model_path": versioned_model_path,
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "feature_columns": feature_cols,
        "metrics": {"mae": mae, "rmse": rmse, "mape": mape},
    }

    return {
        "model_path": SELL_TRAIN_MODEL_PATH,
        "versioned_model_path": versioned_model_path,
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "metrics": selected_metrics,
    }


@router.get("/predict/sell", response_model=SellPredictResponse)
def predict_sell(limit: int = 100) -> SellPredictResponse:
    artifact = load_artifact(SELL_MODEL_PATH)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Sell model not found: {SELL_MODEL_PATH}")

    best_forecast = artifact.get("best_forecast")
    if isinstance(best_forecast, pd.DataFrame) and not best_forecast.empty:
        forecast_df = best_forecast.copy()
    else:
        forecast_df = forecast_from_artifact_model(artifact, horizon_days=730)

    if forecast_df.empty:
        raise HTTPException(status_code=404, detail="No sell forecast available.")

    rec = compute_sell_recommendation_from_forecast(forecast_df)
    rows = safe_head(forecast_df, n=max(1, min(limit, 5000)))

    return {
        "best_model": artifact.get("best_model"),
        "best_month": rec.get("best_month"),
        "best_date": rec.get("best_date"),
        "best_value": rec.get("best_value"),
        "rows": rows,
        "note": "Prediction generated from the saved sell model.",
    }


@router.post("/lookup/sell-monthly/save", response_model=SellMonthlySaveResponse)
def save_sell_monthly() -> SellMonthlySaveResponse:
    ml_schema = os.getenv("ML_SCHEMA", "ml")
    result_table = os.getenv("ML_SELL_TABLE", "best_time_to_sell_monthly")

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ml_schema or ""):
        raise HTTPException(status_code=400, detail=f"Invalid ML_SCHEMA={ml_schema!r}")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", result_table or ""):
        raise HTTPException(status_code=400, detail=f"Invalid ML_SELL_TABLE={result_table!r}")

    artifact = load_artifact(SELL_MODEL_PATH)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Sell model not found: {SELL_MODEL_PATH}")

    monthly_table = artifact.get("monthly_table")
    if not isinstance(monthly_table, pd.DataFrame) or monthly_table.empty:
        generated = forecast_from_artifact_model(artifact, horizon_days=730)
        if generated.empty:
            raise HTTPException(status_code=404, detail="No sell forecast available to save.")
        generated = generated.copy()
        generated["month"] = pd.to_datetime(generated["ds"]).dt.to_period("M").astype(str)
        monthly_table = generated.groupby("month", as_index=False)["yhat"].sum().rename(columns={"yhat": "expected_revenue_sum"})

    out = monthly_table.copy()
    if "month" not in out.columns:
        out = out.reset_index().rename(columns={out.columns[0]: "month"})
    out["month"] = out["month"].astype(str)

    val_col = None
    for c in ["expected_revenue_sum", "yhat_sum", "expected_ca_sum"]:
        if c in out.columns:
            val_col = c
            break
    if val_col is None and len(out.columns) >= 2:
        val_col = out.columns[1]
    out["expected_revenue_sum"] = pd.to_numeric(out[val_col], errors="coerce") if val_col else np.nan

    out["period_start"] = pd.to_datetime(out["month"] + "-01", errors="coerce")
    out["period_end"] = out["period_start"] + pd.offsets.MonthEnd(0)
    out["period_start"] = out["period_start"].dt.date
    out["period_end"] = out["period_end"].dt.date

    best_month_txt = str(artifact.get("best_month") or "") or None
    best_day_value = artifact.get("best_value")
    best_day = artifact.get("best_date")
    model_name = artifact.get("best_model")

    out["is_best_month"] = out["month"].eq(best_month_txt) if best_month_txt else False
    out["best_month"] = best_month_txt
    out["best_day"] = pd.to_datetime(best_day, errors="coerce").date() if best_day is not None else None
    out["best_day_value"] = float(best_day_value) if best_day_value is not None else None
    out["model"] = str(model_name) if model_name is not None else None

    keep = [
        "month",
        "period_start",
        "period_end",
        "expected_revenue_sum",
        "is_best_month",
        "best_month",
        "best_day",
        "best_day_value",
        "model",
    ]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan
    out = out[keep].copy()

    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    out.insert(0, "created_at", created_at)
    out.insert(0, "run_id", run_id)

    engine = _build_pg_engine_from_env()
    schema_q = _qident_local(ml_schema)
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_q}"))

    out.to_sql(result_table, engine, schema=ml_schema, if_exists="append", index=False, method="multi", chunksize=2000)

    return {
        "saved_rows": int(len(out)),
        "run_id": run_id,
        "target_table": f"{ml_schema}.{result_table}",
    }


