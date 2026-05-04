from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

LOGGER_NAME = "sougui.observability"
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

MODEL_METRIC = Gauge("ml_model_metric_value", "Model metric value", ["model", "metric"])
DATA_METRIC = Gauge("ml_data_metric_value", "Data metric value", ["model", "metric"])
DRIFT_METRIC = Gauge("ml_drift_metric_value", "Drift metric value", ["model", "metric"])
RETRAINING_TRIGGERS_TOTAL = Counter("ml_retraining_triggers_total", "Retraining triggers", ["model", "reason"])


def setup_metrics(app: FastAPI) -> None:
    if getattr(app.state, "_sougui_metrics_ready", False):
        return
    Instrumentator(should_group_status_codes=True, should_ignore_untemplated=True).instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=False,
    )
    app.state._sougui_metrics_ready = True


def log_event(level: int, event: str, **payload: Any) -> None:
    record = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    logger.log(level, json.dumps(record, default=_json_default, ensure_ascii=True))


def record_error(model: str, endpoint: str, error_type: str, **payload: Any) -> None:
    log_event(logging.ERROR, "api_error", model=model, endpoint=endpoint, error_type=error_type, **payload)


def record_retraining_trigger(model: str, reason: str, **payload: Any) -> None:
    RETRAINING_TRIGGERS_TOTAL.labels(model=model, reason=reason).inc()
    log_event(logging.INFO, "retraining_trigger", model=model, reason=reason, **payload)


def set_model_metric(model: str, metric: str, value: Any) -> None:
    if value is None:
        return
    try:
        numeric_value = float(value)
    except Exception:
        return
    if math.isnan(numeric_value) or math.isinf(numeric_value):
        return
    MODEL_METRIC.labels(model=model, metric=metric).set(numeric_value)


def set_data_metric(model: str, metric: str, value: Any) -> None:
    if value is None:
        return
    try:
        numeric_value = float(value)
    except Exception:
        return
    if math.isnan(numeric_value) or math.isinf(numeric_value):
        return
    DATA_METRIC.labels(model=model, metric=metric).set(numeric_value)


def set_drift_metric(model: str, metric: str, value: Any) -> None:
    if value is None:
        return
    try:
        numeric_value = float(value)
    except Exception:
        return
    if math.isnan(numeric_value) or math.isinf(numeric_value):
        return
    DRIFT_METRIC.labels(model=model, metric=metric).set(numeric_value)


def publish_model_metrics(model: str, metrics: dict[str, Any]) -> None:
    for metric_name, metric_value in metrics.items():
        set_model_metric(model, metric_name, metric_value)


def publish_drift_metrics(model: str, metrics: dict[str, Any]) -> None:
    for metric_name, metric_value in metrics.items():
        set_drift_metric(model, metric_name, metric_value)


def compute_missing_ratio(df: pd.DataFrame | None) -> float | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    total_cells = int(df.shape[0] * df.shape[1])
    if total_cells <= 0:
        return None
    missing_cells = int(df.isna().sum().sum())
    return float(missing_cells / total_cells)


def compute_data_freshness_seconds(df: pd.DataFrame | None, source_path: str | Path | None = None) -> float | None:
    if isinstance(source_path, str):
        source_path = Path(source_path)

    timestamps: list[pd.Timestamp] = []
    if isinstance(df, pd.DataFrame) and not df.empty:
        candidate_columns = [
            col
            for col in df.columns
            if any(token in col.lower() for token in ("date", "time", "timestamp", "ds", "created_at", "updated_at"))
        ]
        for col in candidate_columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if hasattr(parsed, "dropna"):
                parsed = parsed.dropna()
            if len(parsed) > 0:
                timestamps.append(pd.Timestamp(parsed.max()))

    if not timestamps and isinstance(source_path, Path) and source_path.exists():
        timestamps.append(pd.Timestamp(source_path.stat().st_mtime, unit="s", tz="UTC"))

    if not timestamps:
        return None

    latest = max(timestamps)
    now = pd.Timestamp.now(tz="UTC")
    freshness = (now - latest.tz_convert("UTC") if latest.tzinfo is not None else now.tz_localize(None) - latest.tz_localize(None))
    return float(max(freshness.total_seconds(), 0.0))


def publish_data_metrics(model: str, df: pd.DataFrame | None, source_path: str | Path | None = None) -> None:
    missing_ratio = compute_missing_ratio(df)
    freshness_seconds = compute_data_freshness_seconds(df, source_path=source_path)
    if missing_ratio is not None:
        set_data_metric(model, "missing_ratio", missing_ratio)
    if freshness_seconds is not None:
        set_data_metric(model, "freshness_seconds", freshness_seconds)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
