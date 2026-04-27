import os
import re
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import create_engine, text

from api_common import (
    PROMOTE_MODEL_PATH,
    compute_promote_monthly_from_forecast,
    compute_promote_recommendation_from_forecast,
    load_artifact,
    load_dataset_from_sources,
    resolve_output_path,
    save_versioned_artifact,
    safe_head,
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PROMOTE_EXPERIMENT_NAME = os.getenv("MLFLOW_PROMOTE_EXPERIMENT_NAME", "promote-forecasting")
PROMOTE_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_PROMOTE_MODEL_NAME", "promote-demand-forecaster")

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    MlflowClient = None
    MLFLOW_AVAILABLE = False

router = APIRouter(tags=["promote"])

PROMOTE_TRAIN_MODEL_PATH = os.getenv("PROMOTE_TRAIN_MODEL_PATH", PROMOTE_MODEL_PATH)
PROMOTE_NOTEBOOK_DATASET_PATH = os.getenv("PROMOTE_NOTEBOOK_DATASET_PATH", "promote_feature_dataset.parquet")
PROMOTE_DB_DATASET_TABLE = os.getenv("ML_PROMOTE_DATASET_TABLE", "promote")
_promote_prepared_dataset: pd.DataFrame | None = None
_promote_train_results: dict[str, Any] | None = None

RAMADAN_CALENDAR: dict[int, tuple[str, str, str, str]] = {
    2024: ("2024-03-11", "2024-04-09", "2024-04-10", "2024-04-13"),
    2025: ("2025-03-01", "2025-03-29", "2025-03-30", "2025-04-02"),
    2026: ("2026-02-18", "2026-03-19", "2026-03-20", "2026-03-23"),
    2027: ("2027-02-08", "2027-03-09", "2027-03-10", "2027-03-13"),
}


class PromoteRecommendResponse(BaseModel):
    best_model: str | None = None
    best_month: str | None = None
    best_day: str | None = None
    best_day_value: float | None = None
    note: str | None = None


class PromotePlanResponse(BaseModel):
    best_model: str | None = None
    monthly_2027: list[dict[str, Any]]
    promo_plan_2027: list[dict[str, Any]]
    note: str | None = None


class PromotePrepareDatasetResponse(BaseModel):
    rows: int
    feature_columns: list[str]
    preview: list[dict[str, Any]]


class PromoteTrainResponse(BaseModel):
    model_path: str
    versioned_model_path: str | None = None
    rows_train: int
    rows_test: int
    best_model: str
    metrics: dict[str, float]
    candidate_metrics: dict[str, dict[str, float]] | None = None
    mlflow: dict[str, Any] | None = None


class PromoteTrainResultsResponse(BaseModel):
    available: bool
    results: dict[str, Any] | None = None


class PromoteMonthlySaveResponse(BaseModel):
    saved_rows: int
    run_id: str
    target_table: str


class PromotePredictResponse(BaseModel):
    best_model: str | None = None
    best_month: str | None = None
    best_day: str | None = None
    best_day_value: float | None = None
    monthly_2027: list[dict[str, Any]]
    promo_plan_2027: list[dict[str, Any]]
    note: str | None = None


class PromoteFormPredictRequest(BaseModel):
    date: date
    last_day_revenue: float = Field(ge=0.0)
    last_week_revenue: float = Field(ge=0.0)
    two_weeks_revenue: float = Field(ge=0.0)
    rolling_7d_revenue: float = Field(ge=0.0)
    planned_discount_pct: float = Field(ge=0.0, le=90.0)
    ramadan_days: int = Field(default=0, ge=0, le=30)
    eid_window_days: int = Field(default=0, ge=0, le=10)


class PromoteFormPredictResponse(BaseModel):
    best_model: str | None = None
    model_version: str | None = None
    decision: str
    demand_level: str
    predicted_revenue: float
    baseline_revenue: float
    uplift_pct: float
    recommended_discount_pct: float
    recommended_discount_display: str
    confidence_score: float
    mlflow: dict[str, Any] | None = None
    inputs: dict[str, Any]
    reasoning: list[str]


def _default_monthly_discount(month: int) -> float:
    base = {
        1: 0.18,
        2: 0.19,
        3: 0.28,
        4: 0.33,
        5: 0.24,
        6: 0.17,
        7: 0.16,
        8: 0.20,
        9: 0.21,
        10: 0.26,
        11: 0.22,
        12: 0.15,
    }
    return float(base.get(int(month), 0.2))


def _event_context(ts: pd.Timestamp) -> dict[str, int]:
    info = {"ramadan_days": 0, "eid_window_days": 0}
    calendar = RAMADAN_CALENDAR.get(int(ts.year))
    if calendar is None:
        return info

    ramadan_start, ramadan_end, eid_start, eid_end = [pd.Timestamp(value) for value in calendar]
    day = pd.Timestamp(ts).normalize()
    if ramadan_start <= day <= ramadan_end:
        info["ramadan_days"] = int((ramadan_end - day).days + 1)
    if eid_start <= day <= eid_end:
        info["eid_window_days"] = int((eid_end - day).days + 1)
    return info


def _apply_event_features(df: pd.DataFrame, date_col: str = "ds") -> pd.DataFrame:
    work = df.copy()
    dates = pd.to_datetime(work[date_col], errors="coerce")
    event_rows = [_event_context(ts) if not pd.isna(ts) else {"ramadan_days": 0, "eid_window_days": 0} for ts in dates]
    event_df = pd.DataFrame(event_rows, index=work.index)
    for col in ["ramadan_days", "eid_window_days"]:
        if col not in work.columns:
            work[col] = event_df[col]
        else:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(event_df[col]).fillna(0).astype(int)
    return work


def _default_promote_training_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", "2026-12-31", freq="D")
    month_offsets = {
        1: 400,
        2: 250,
        3: -800,
        4: -1200,
        5: -250,
        6: 500,
        7: 650,
        8: 150,
        9: -200,
        10: -700,
        11: -150,
        12: 1100,
    }
    rows: list[dict[str, Any]] = []
    for ts in dates:
        event = _event_context(pd.Timestamp(ts))
        seasonal = 10800 + 1400 * np.sin(2 * np.pi * (ts.dayofyear / 365.25))
        seasonal += month_offsets.get(int(ts.month), 0)
        weekend = 900 if ts.weekday() in {4, 5} else (-220 if ts.weekday() == 0 else 0)
        discount_pct = np.clip(
            _default_monthly_discount(int(ts.month))
            + (0.05 if event["ramadan_days"] > 0 else 0.0)
            + (0.04 if event["eid_window_days"] > 0 else 0.0)
            + rng.normal(0.0, 0.025),
            0.05,
            0.45,
        )
        weak_demand = -1100 * min(event["ramadan_days"] / 30.0, 1.0)
        holiday_push = 700 * min(event["eid_window_days"] / 4.0, 1.0)
        promo_lift = (8500 * discount_pct) + (2600 * discount_pct * min(event["ramadan_days"] / 30.0 + 1.0, 1.6))
        noise = rng.normal(0.0, 520.0)
        revenue = max(2400.0, seasonal + weekend + weak_demand + holiday_push + promo_lift + noise)
        rows.append(
            {
                "date": ts.date().isoformat(),
                "revenue": round(revenue, 2),
                "discount_pct": round(float(discount_pct), 4),
                "ramadan_days": event["ramadan_days"],
                "eid_window_days": event["eid_window_days"],
            }
        )
    return pd.DataFrame(rows)


def _prepare_promote_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "date" not in work.columns and "ds" not in work.columns:
        raise HTTPException(status_code=400, detail="Missing date column. Provide 'date' or 'ds'.")
    if "revenue" not in work.columns and "y" not in work.columns:
        raise HTTPException(status_code=400, detail="Missing revenue column. Provide 'revenue' or 'y'.")

    date_col = "date" if "date" in work.columns else "ds"
    y_col = "revenue" if "revenue" in work.columns else "y"

    work["ds"] = pd.to_datetime(work[date_col], errors="coerce")
    work["y"] = pd.to_numeric(work[y_col], errors="coerce")
    work = _apply_event_features(work, date_col="ds")

    if "discount_pct" in work.columns:
        work["discount_pct"] = pd.to_numeric(work["discount_pct"], errors="coerce").fillna(0.0)
    elif "remise" in work.columns:
        work["discount_pct"] = pd.to_numeric(work["remise"], errors="coerce").fillna(0.0)
    else:
        work["discount_pct"] = 0.0

    if work["discount_pct"].max() > 1.5:
        work["discount_pct"] = (work["discount_pct"] / 100.0).clip(0.0, 0.9)

    work = work.dropna(subset=["ds", "y"]).copy()
    if work.empty:
        raise HTTPException(status_code=400, detail="No valid rows after parsing date/revenue.")

    aggregate_map: dict[str, tuple[str, str]] = {
        "y": ("y", "sum"),
        "discount_pct": ("discount_pct", "mean"),
        "ramadan_days": ("ramadan_days", "max"),
        "eid_window_days": ("eid_window_days", "max"),
    }
    daily = work.groupby("ds", as_index=False).agg(**aggregate_map).sort_values("ds")
    daily["dow"] = daily["ds"].dt.weekday
    daily["month"] = daily["ds"].dt.month
    daily["day"] = daily["ds"].dt.day
    daily["is_weekend"] = daily["dow"].isin({5, 6}).astype(int)
    daily["lag_1"] = daily["y"].shift(1)
    daily["lag_7"] = daily["y"].shift(7)
    daily["lag_14"] = daily["y"].shift(14)
    daily["roll_mean_7"] = daily["y"].shift(1).rolling(7).mean()
    daily = daily.dropna().reset_index(drop=True)

    if daily.empty:
        raise HTTPException(status_code=400, detail="Not enough history to create lag features.")

    return daily


def _load_promote_training_dataset() -> pd.DataFrame:
    global _promote_prepared_dataset

    ds = load_dataset_from_sources(_promote_prepared_dataset, PROMOTE_DB_DATASET_TABLE, PROMOTE_NOTEBOOK_DATASET_PATH)
    if ds is None or ds.empty:
        ds = _default_promote_training_dataset()

    work = ds.copy()
    if {"ds", "y", "lag_1", "lag_7", "lag_14", "roll_mean_7"}.issubset(set(work.columns)):
        work["ds"] = pd.to_datetime(work["ds"], errors="coerce")
        work = work.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
        work = _apply_event_features(work, date_col="ds")
        if "discount_pct" not in work.columns:
            work["discount_pct"] = _default_monthly_discount(1)
        work["discount_pct"] = pd.to_numeric(work["discount_pct"], errors="coerce").fillna(0.0)
        work["dow"] = pd.to_numeric(work.get("dow", work["ds"].dt.weekday), errors="coerce").fillna(work["ds"].dt.weekday)
        work["month"] = pd.to_numeric(work.get("month", work["ds"].dt.month), errors="coerce").fillna(work["ds"].dt.month)
        work["day"] = pd.to_numeric(work.get("day", work["ds"].dt.day), errors="coerce").fillna(work["ds"].dt.day)
        work["is_weekend"] = pd.to_numeric(
            work.get("is_weekend", work["ds"].dt.weekday.isin({5, 6}).astype(int)),
            errors="coerce",
        ).fillna(0)
        _promote_prepared_dataset = work
        return work

    prepared = _prepare_promote_features(work)
    _promote_prepared_dataset = prepared
    return prepared


def _candidate_promote_models(random_state: int) -> dict[str, Any]:
    return {
        "extra_trees": ExtraTreesRegressor(n_estimators=400, random_state=random_state, min_samples_leaf=2),
        "random_forest": RandomForestRegressor(n_estimators=320, random_state=random_state, min_samples_leaf=2),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state, n_estimators=250, learning_rate=0.05),
    }


def _metric_payload(y_true, y_pred) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true_array, y_pred_array))
    rmse = float(np.sqrt(mean_squared_error(y_true_array, y_pred_array)))
    mask = np.abs(y_true_array) > 1e-9
    mape = (
        float(np.mean(np.abs((y_true_array[mask] - y_pred_array[mask]) / y_true_array[mask])) * 100.0)
        if mask.any()
        else float("nan")
    )
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _model_rank_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    mape = float(metrics.get("mape", float("inf")))
    if not np.isfinite(mape):
        mape = float("inf")
    return (mape, float(metrics.get("rmse", float("inf"))), float(metrics.get("mae", float("inf"))))


def _predict_promote_value(model, feature_columns: list[str], feature_row: dict[str, Any]) -> float:
    frame = pd.DataFrame([feature_row])
    for col in feature_columns:
        if col not in frame.columns:
            frame[col] = 0.0
    frame = frame.reindex(columns=feature_columns, fill_value=0.0)
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return float(model.predict(frame)[0])


def _build_promote_future_plan(model, feature_columns: list[str], history_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history = history_frame.copy().sort_values("ds").reset_index(drop=True)
    if len(history) < 14:
        raise HTTPException(status_code=400, detail="Promote history must contain at least 14 rows to build a forecast.")

    history_values = history["y"].astype(float).tolist()
    monthly_history = history.assign(month=history["ds"].dt.month).groupby("month", as_index=False)["y"].mean()
    month_min = float(monthly_history["y"].min()) if not monthly_history.empty else 0.0
    month_max = float(monthly_history["y"].max()) if not monthly_history.empty else 1.0
    month_strength: dict[int, float] = {}
    for row in monthly_history.itertuples(index=False):
        if abs(month_max - month_min) < 1e-9:
            month_strength[int(row.month)] = 0.5
        else:
            month_strength[int(row.month)] = float((row.y - month_min) / (month_max - month_min))

    records: list[dict[str, Any]] = []
    start_day = pd.Timestamp("2027-01-01")
    end_day = pd.Timestamp("2027-12-31")
    for ts in pd.date_range(start_day, end_day, freq="D"):
        last_seven = history_values[-7:]
        event = _event_context(ts)
        month_bias = 1.0 - month_strength.get(int(ts.month), 0.5)
        planned_discount = np.clip(
            _default_monthly_discount(int(ts.month))
            + 0.10 * month_bias
            + (0.05 if event["ramadan_days"] > 0 else 0.0)
            + (0.04 if event["eid_window_days"] > 0 else 0.0),
            0.08,
            0.45,
        )
        feature_row = {
            "discount_pct": planned_discount,
            "ramadan_days": event["ramadan_days"],
            "eid_window_days": event["eid_window_days"],
            "dow": ts.weekday(),
            "month": ts.month,
            "day": ts.day,
            "is_weekend": int(ts.weekday() in {5, 6}),
            "lag_1": history_values[-1],
            "lag_7": history_values[-7],
            "lag_14": history_values[-14],
            "roll_mean_7": float(np.mean(last_seven)),
        }
        yhat = _predict_promote_value(model, feature_columns, feature_row)
        yhat = max(0.0, float(yhat))
        records.append(
            {
                "ds": ts,
                "yhat": yhat,
                "discount_pct": planned_discount,
                "ramadan_days": event["ramadan_days"],
                "eid_window_days": event["eid_window_days"],
            }
        )
        history_values.append(yhat)

    future_daily = pd.DataFrame(records)
    monthly = compute_promote_monthly_from_forecast(future_daily[["ds", "yhat"]])
    monthly_events = (
        future_daily.assign(month=future_daily["ds"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(
            ramadan_days=("ramadan_days", "sum"),
            eid_window_days=("eid_window_days", "sum"),
            planned_discount_pct=("discount_pct", "mean"),
        )
    )
    monthly = monthly.merge(monthly_events, on="month", how="left")
    monthly["promotion_pct"] = np.clip(
        np.maximum(monthly["promotion_pct"].astype(float), monthly["planned_discount_pct"].astype(float)),
        0.10,
        0.50,
    )
    monthly["promotion_display"] = (monthly["promotion_pct"] * 100).round(1).astype(str) + "%"
    promo_plan = monthly[
        ["month", "promotion_pct", "promotion_display", "ramadan_days", "eid_window_days"]
    ].copy()
    promo_plan["action"] = promo_plan.apply(
        lambda row: f"Apply {row['promotion_display']} discount" + (" with event push" if row["ramadan_days"] or row["eid_window_days"] else ""),
        axis=1,
    )
    return future_daily, monthly, promo_plan


def _serialize_candidate_metrics(candidate_metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for model_name, metrics in candidate_metrics.items():
        out[model_name] = {key: float(value) for key, value in metrics.items() if np.isfinite(value)}
    return out


def _log_promote_mlflow_run(
    *,
    run_name: str,
    model_name: str,
    model,
    feature_columns: list[str],
    random_state: int,
    rows_train: int,
    rows_test: int,
    metrics: dict[str, float],
    comparison_label: str,
    register_model: bool = False,
) -> dict[str, Any] | None:
    if not MLFLOW_AVAILABLE:
        return None

    artifact_dir = Path("./mlartifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{run_name}.joblib"
    payload = {
        "task": "promote_train_api",
        "model_name": model_name,
        "model": model,
        "feature_columns": feature_columns,
        "random_state": int(random_state),
    }
    joblib.dump(payload, artifact_path)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(PROMOTE_EXPERIMENT_NAME)
    client = MlflowClient() if MlflowClient is not None else None
    info: dict[str, Any] = {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_name": PROMOTE_EXPERIMENT_NAME,
        "registered_model_name": None,
        "registered_model_version": None,
    }
    with mlflow.start_run(run_name=run_name) as run:
        info["run_id"] = run.info.run_id
        mlflow.set_tags({"task": "promote_train_api", "model_name": model_name, "comparison_label": comparison_label})
        mlflow.log_params(
            {
                "random_state": int(random_state),
                "rows_train": int(rows_train),
                "rows_test": int(rows_test),
                "feature_count": int(len(feature_columns)),
                "comparison_label": comparison_label,
                "model_name": model_name,
            }
        )
        mlflow.log_metrics({key: float(value) for key, value in metrics.items() if np.isfinite(value)})
        mlflow.log_artifact(str(artifact_path), artifact_path="artifacts")

        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
            info["model_uri"] = f"runs:/{run.info.run_id}/model"
        except Exception:
            info["model_uri"] = None

        if register_model and info.get("model_uri") and client is not None:
            try:
                client.get_registered_model(PROMOTE_REGISTERED_MODEL_NAME)
            except Exception:
                try:
                    client.create_registered_model(PROMOTE_REGISTERED_MODEL_NAME)
                except Exception:
                    pass
            try:
                model_version = mlflow.register_model(model_uri=info["model_uri"], name=PROMOTE_REGISTERED_MODEL_NAME)
                info["registered_model_name"] = PROMOTE_REGISTERED_MODEL_NAME
                info["registered_model_version"] = str(getattr(model_version, "version", None) or "")
            except Exception:
                pass
    return info


def _train_and_save_promote_artifact(random_state: int = 42) -> tuple[dict[str, Any], dict[str, Any]]:
    global _promote_train_results

    ds = _load_promote_training_dataset()
    ds = ds.copy().sort_values("ds").reset_index(drop=True)
    feature_cols = [c for c in ds.columns if c not in ["ds", "y"]]
    split_idx = max(14, int(len(ds) * 0.8))
    train_df = ds.iloc[:split_idx]
    test_df = ds.iloc[split_idx:]
    if test_df.empty:
        raise HTTPException(status_code=400, detail="Not enough rows to create train/test split.")

    X_train = train_df[feature_cols]
    y_train = train_df["y"]
    X_test = test_df[feature_cols]
    y_test = test_df["y"]

    candidate_models = _candidate_promote_models(random_state=random_state)
    fitted_models: dict[str, Any] = {}
    candidate_metrics: dict[str, dict[str, float]] = {}
    candidate_mlflow: dict[str, Any] = {}
    for model_name, model in candidate_models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics = _metric_payload(y_test, prediction)
        fitted_models[model_name] = model
        candidate_metrics[model_name] = metrics
        candidate_mlflow[model_name] = _log_promote_mlflow_run(
            run_name=f"promote_{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            model_name=model_name,
            model=model,
            feature_columns=feature_cols,
            random_state=random_state,
            rows_train=len(train_df),
            rows_test=len(test_df),
            metrics=metrics,
            comparison_label="candidate",
            register_model=False,
        )

    best_model_name = min(candidate_metrics.keys(), key=lambda item: _model_rank_key(candidate_metrics[item]))
    best_model = fitted_models[best_model_name]
    best_metrics = candidate_metrics[best_model_name]

    future_daily, monthly_2027, promo_plan_2027 = _build_promote_future_plan(best_model, feature_cols, ds[["ds", "y"]])
    best_recommendation = compute_promote_recommendation_from_forecast(future_daily[["ds", "yhat"]])
    best_mlflow = _log_promote_mlflow_run(
        run_name=f"promote_best_{best_model_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        model_name=best_model_name,
        model=best_model,
        feature_columns=feature_cols,
        random_state=random_state,
        rows_train=len(train_df),
        rows_test=len(test_df),
        metrics=best_metrics,
        comparison_label="selected",
        register_model=True,
    )

    artifact = {
        "task": "promote_train_api",
        "best_model": best_model_name,
        "model": best_model,
        "feature_columns": feature_cols,
        "monthly_2027": monthly_2027,
        "promo_plan_2027": promo_plan_2027,
        "best_month": best_recommendation.get("best_month"),
        "best_day": best_recommendation.get("best_day"),
        "best_day_value": best_recommendation.get("best_day_value"),
        "history": ds[["ds", "y"]].copy(),
        "metrics": best_metrics,
        "candidate_metrics": _serialize_candidate_metrics(candidate_metrics),
        "mlflow": {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment_name": PROMOTE_EXPERIMENT_NAME,
            "best_run": best_mlflow,
            "candidate_runs": candidate_mlflow,
        },
    }
    summary = {
        "model_path": PROMOTE_MODEL_PATH,
        "versioned_model_path": None,
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "best_model": best_model_name,
        "metrics": {key: float(value) for key, value in best_metrics.items() if np.isfinite(value)},
        "candidate_metrics": artifact["candidate_metrics"],
        "mlflow": artifact["mlflow"],
    }
    artifact["training_summary"] = summary
    versioned_model_path = save_versioned_artifact(artifact, PROMOTE_MODEL_PATH)
    summary["versioned_model_path"] = versioned_model_path
    artifact["training_summary"] = summary
    joblib.dump(artifact, resolve_output_path(PROMOTE_MODEL_PATH))
    if PROMOTE_TRAIN_MODEL_PATH != PROMOTE_MODEL_PATH:
        save_versioned_artifact(artifact, PROMOTE_TRAIN_MODEL_PATH)
    _promote_train_results = summary
    return artifact, summary


def _is_valid_promote_artifact(artifact: dict[str, Any] | None) -> bool:
    if not isinstance(artifact, dict):
        return False
    model = artifact.get("model")
    feature_columns = artifact.get("feature_columns")
    monthly_2027 = artifact.get("monthly_2027")
    return model is not None and isinstance(feature_columns, list) and len(feature_columns) > 0 and isinstance(monthly_2027, pd.DataFrame)


def ensure_promote_prediction_artifact() -> dict[str, Any]:
    artifact = load_artifact(PROMOTE_MODEL_PATH)
    if _is_valid_promote_artifact(artifact):
        return artifact
    artifact, _ = _train_and_save_promote_artifact(random_state=42)
    return artifact


def _promote_confidence_score(artifact: dict[str, Any]) -> float:
    metrics = artifact.get("metrics") if isinstance(artifact.get("metrics"), dict) else {}
    mape = float(metrics.get("mape", 35.0)) if metrics else 35.0
    score = 1.0 - min(max(mape, 0.0), 80.0) / 100.0
    return float(np.clip(score, 0.2, 0.98))


def _promote_training_results_from_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    summary = artifact.get("training_summary") if isinstance(artifact.get("training_summary"), dict) else None
    if summary:
        return summary
    return {
        "model_path": PROMOTE_MODEL_PATH,
        "versioned_model_path": None,
        "rows_train": 0,
        "rows_test": 0,
        "best_model": str(artifact.get("best_model") or "n/a"),
        "metrics": artifact.get("metrics") if isinstance(artifact.get("metrics"), dict) else {},
        "candidate_metrics": artifact.get("candidate_metrics") if isinstance(artifact.get("candidate_metrics"), dict) else {},
        "mlflow": artifact.get("mlflow") if isinstance(artifact.get("mlflow"), dict) else None,
    }


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
        raise HTTPException(status_code=503, detail=f"PostgreSQL configuration is missing. Missing: {', '.join(missing)}")

    return create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
        pool_pre_ping=True,
    )


@router.get("/train/promote/dataset", response_model=PromotePrepareDatasetResponse)
def get_promote_dataset(limit: int = 20) -> PromotePrepareDatasetResponse:
    n = max(1, min(limit, 5000))
    ds = _load_promote_training_dataset()
    feature_cols = [c for c in ds.columns if c not in ["ds", "y"]]
    return {
        "rows": int(len(ds)),
        "feature_columns": feature_cols,
        "preview": safe_head(ds, n=n),
    }


@router.post("/train/promote/train", response_model=PromoteTrainResponse)
def train_promote_model(random_state: int = 42) -> PromoteTrainResponse:
    _, summary = _train_and_save_promote_artifact(random_state=random_state)
    return summary


@router.get("/train/promote/results", response_model=PromoteTrainResultsResponse)
def get_promote_train_results() -> PromoteTrainResultsResponse:
    global _promote_train_results
    if isinstance(_promote_train_results, dict):
        return {"available": True, "results": _promote_train_results}

    artifact = load_artifact(PROMOTE_MODEL_PATH)
    if _is_valid_promote_artifact(artifact):
        summary = _promote_training_results_from_artifact(artifact)
        _promote_train_results = summary
        return {"available": True, "results": summary}

    return {"available": False, "results": None}


@router.get("/predict/promote", response_model=PromotePredictResponse)
def predict_promote(limit: int = 100) -> PromotePredictResponse:
    artifact = ensure_promote_prediction_artifact()
    monthly_2027 = artifact.get("monthly_2027")
    promo_plan_2027 = artifact.get("promo_plan_2027")
    if not isinstance(monthly_2027, pd.DataFrame) or monthly_2027.empty:
        raise HTTPException(status_code=404, detail="No promote forecast available.")
    if not isinstance(promo_plan_2027, pd.DataFrame):
        promo_plan_2027 = pd.DataFrame()
    return {
        "best_model": artifact.get("best_model"),
        "best_month": artifact.get("best_month"),
        "best_day": artifact.get("best_day"),
        "best_day_value": artifact.get("best_day_value"),
        "monthly_2027": safe_head(monthly_2027, n=max(1, min(limit, 5000))),
        "promo_plan_2027": safe_head(promo_plan_2027, n=max(1, min(limit, 5000))),
        "note": "Prediction generated from the promote forecasting model.",
    }


@router.post("/predict/promote/form", response_model=PromoteFormPredictResponse)
def predict_promote_from_form(payload: PromoteFormPredictRequest) -> PromoteFormPredictResponse:
    artifact = ensure_promote_prediction_artifact()
    model = artifact.get("model")
    feature_columns = artifact.get("feature_columns")
    if model is None or not isinstance(feature_columns, list) or not feature_columns:
        raise HTTPException(status_code=500, detail="Promote artifact is missing model metadata.")

    target_day = pd.Timestamp(payload.date)
    planned_discount_ratio = float(payload.planned_discount_pct) / 100.0
    feature_row = {
        "discount_pct": planned_discount_ratio,
        "ramadan_days": int(payload.ramadan_days),
        "eid_window_days": int(payload.eid_window_days),
        "dow": int(target_day.weekday()),
        "month": int(target_day.month),
        "day": int(target_day.day),
        "is_weekend": int(target_day.weekday() in {5, 6}),
        "lag_1": float(payload.last_day_revenue),
        "lag_7": float(payload.last_week_revenue),
        "lag_14": float(payload.two_weeks_revenue),
        "roll_mean_7": float(payload.rolling_7d_revenue),
    }

    predicted_revenue = max(0.0, _predict_promote_value(model, feature_columns, feature_row))
    baseline_row = feature_row.copy()
    baseline_row["discount_pct"] = 0.0
    model_baseline = max(0.0, _predict_promote_value(model, feature_columns, baseline_row))
    recent_baseline = max(
        float(payload.rolling_7d_revenue),
        float(payload.last_day_revenue),
        float(payload.last_week_revenue),
        float(payload.two_weeks_revenue),
        1.0,
    )
    baseline_revenue = max(model_baseline, recent_baseline * 0.85, 1.0)
    uplift_pct = float((predicted_revenue - baseline_revenue) / baseline_revenue)

    demand_pressure = max(0.0, (recent_baseline - predicted_revenue) / max(recent_baseline, 1.0))
    event_intensity = min(1.0, (payload.ramadan_days / 30.0) * 0.65 + (payload.eid_window_days / 10.0) * 0.35)
    promo_need = float(np.clip((0.55 * demand_pressure) + (0.25 * event_intensity) + (0.20 * max(0.0, -uplift_pct)), 0.0, 1.0))
    recommended_discount_pct = float(np.clip(10.0 + (35.0 * promo_need), 8.0, 45.0))

    demand_ratio = predicted_revenue / max(recent_baseline, 1.0)
    if demand_ratio >= 1.08:
        demand_level = "forte"
    elif demand_ratio >= 0.96:
        demand_level = "stable"
    elif demand_ratio >= 0.85:
        demand_level = "fragile"
    else:
        demand_level = "faible"

    if promo_need >= 0.7 or (event_intensity >= 0.45 and uplift_pct >= 0.04):
        decision = "Promotion forte recommandee"
    elif promo_need >= 0.45 or uplift_pct >= 0.05:
        decision = "Promotion moderee recommandee"
    elif uplift_pct >= 0.0:
        decision = "Promotion legere suffisante"
    else:
        decision = "Promotion non prioritaire"

    reasoning = [
        f"Demande projetee: {demand_level} par rapport a la moyenne recente.",
        f"Impact estime de la remise: {uplift_pct * 100:.1f}% sur le chiffre d'affaires journalier.",
        f"Contexte evenementiel: {payload.ramadan_days} jour(s) Ramadan restant(s), {payload.eid_window_days} jour(s) de fenetre Eid.",
    ]

    mlflow_info = artifact.get("mlflow") if isinstance(artifact.get("mlflow"), dict) else None
    best_run = mlflow_info.get("best_run") if isinstance(mlflow_info, dict) and isinstance(mlflow_info.get("best_run"), dict) else None
    return {
        "best_model": artifact.get("best_model"),
        "model_version": str(best_run.get("registered_model_version")) if best_run and best_run.get("registered_model_version") else None,
        "decision": decision,
        "demand_level": demand_level,
        "predicted_revenue": float(round(predicted_revenue, 2)),
        "baseline_revenue": float(round(baseline_revenue, 2)),
        "uplift_pct": float(round(uplift_pct, 4)),
        "recommended_discount_pct": float(round(recommended_discount_pct, 2)),
        "recommended_discount_display": f"{recommended_discount_pct:.1f}%",
        "confidence_score": float(round(_promote_confidence_score(artifact), 4)),
        "mlflow": mlflow_info,
        "inputs": {
            "date": payload.date.isoformat(),
            "last_day_revenue": float(payload.last_day_revenue),
            "last_week_revenue": float(payload.last_week_revenue),
            "two_weeks_revenue": float(payload.two_weeks_revenue),
            "rolling_7d_revenue": float(payload.rolling_7d_revenue),
            "planned_discount_pct": float(payload.planned_discount_pct),
            "ramadan_days": int(payload.ramadan_days),
            "eid_window_days": int(payload.eid_window_days),
        },
        "reasoning": reasoning,
    }


@router.post("/lookup/promote-monthly/save", response_model=PromoteMonthlySaveResponse)
def save_promote_monthly() -> PromoteMonthlySaveResponse:
    ml_schema = os.getenv("ML_SCHEMA", "ml")
    result_table = os.getenv("ML_PROMOTE_TABLE", "best_time_to_promote_monthly")

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ml_schema or ""):
        raise HTTPException(status_code=400, detail=f"Invalid ML_SCHEMA={ml_schema!r}")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", result_table or ""):
        raise HTTPException(status_code=400, detail=f"Invalid ML_PROMOTE_TABLE={result_table!r}")

    artifact = ensure_promote_prediction_artifact()
    monthly_2027 = artifact.get("monthly_2027")
    if not isinstance(monthly_2027, pd.DataFrame) or monthly_2027.empty:
        raise HTTPException(status_code=404, detail="No promote forecast available to save.")

    out = monthly_2027.copy()
    out["month"] = out["month"].astype(str)
    out["period_start"] = pd.to_datetime(out["month"] + "-01", errors="coerce")
    out["period_end"] = out["period_start"] + pd.offsets.MonthEnd(0)
    out["period_start"] = out["period_start"].dt.date
    out["period_end"] = out["period_end"].dt.date

    if "predicted_CA" in out.columns:
        out = out.rename(columns={"predicted_CA": "predicted_ca_sum"})
    if "recommended_promo_pct" in out.columns:
        out = out.rename(columns={"recommended_promo_pct": "promotion_pct"})
    if "recommended_promo_pct_display" in out.columns:
        out = out.rename(columns={"recommended_promo_pct_display": "promotion_display"})

    best_month_txt = str(artifact.get("best_month") or "") or None
    out["is_best_month"] = out["month"].eq(best_month_txt) if best_month_txt else False

    keep = [
        "month",
        "period_start",
        "period_end",
        "predicted_ca_sum",
        "promotion_pct",
        "promotion_display",
        "ramadan_days",
        "eid_window_days",
        "is_best_month",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
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
