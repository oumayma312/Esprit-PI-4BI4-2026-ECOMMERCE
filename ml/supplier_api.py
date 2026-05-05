import logging
import math
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

from api_common import (
    SUPPLIER_MODEL_PATH,
    load_artifact,
    load_ml_dataset_from_db,
    save_versioned_artifact,
    safe_head,
)
from observability import log_event, publish_data_metrics, publish_drift_metrics, publish_model_metrics, record_retraining_trigger, set_model_metric

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

try:
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    MLFLOW_AVAILABLE = False

router = APIRouter(tags=["supplier"])

SUPPLIER_MODEL_NAME = "supplier_clustering"
SUPPLIER_TRAIN_MODEL_PATH = os.getenv("SUPPLIER_TRAIN_MODEL_PATH", SUPPLIER_MODEL_PATH)
SUPPLIER_NOTEBOOK_DATASET_PATH = os.getenv("SUPPLIER_NOTEBOOK_DATASET_PATH", "supplier_feature_dataset.parquet")
SUPPLIER_DB_DATASET_TABLE = os.getenv("ML_SUPPLIER_DATASET_TABLE", "supplier")
_supplier_prepared_dataset: pd.DataFrame | None = None
_supplier_train_results: dict[str, Any] | None = None


class SupplierClusterLookupResponse(BaseModel):
    supplier_name: str
    supplier_id: int | float | str | None = None
    cluster_id: int | None = None
    cluster_name: str | None = None
    note: str | None = None


class SupplierClusterRowResponse(BaseModel):
    supplier_name: str
    cluster_id: int | None = None
    cluster_name: str | None = None


class SupplierClusterListResponse(BaseModel):
    rows: int
    data: list[SupplierClusterRowResponse]
    note: str | None = None


class SupplierClusterSaveResponse(BaseModel):
    saved_rows: int
    run_id: str
    target_table: str


class SupplierPrepareDatasetResponse(BaseModel):
    rows: int
    feature_columns: list[str]
    preview: list[dict[str, Any]]


class SupplierTrainResponse(BaseModel):
    model_path: str
    versioned_model_path: str | None = None
    n_clusters: int
    rows: int
    feature_columns: list[str]
    inertia: float


class SupplierTrainResultsResponse(BaseModel):
    available: bool
    results: dict[str, Any] | None = None


class SupplierDriftResponse(BaseModel):
    drift_detected: bool
    distribution_shift: float
    precision_drop_ratio: float
    confidence_drop_ratio: float
    baseline_silhouette: float | None = None
    current_silhouette: float | None = None
    baseline_confidence: float | None = None
    current_confidence: float | None = None
    note: str | None = None


class SupplierPredictRequest(BaseModel):
    quantity: float | None = None
    unit_price: float | None = None
    total_ht: float | None = None
    total_ttc: float | None = None
    governorate: str | None = None
    city: str | None = None


class SupplierPredictResponse(BaseModel):
    predicted_cluster_id: int
    predicted_cluster_name: str | None = None
    model_path: str
    note: str | None = None


def _native_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def _prepare_supplier_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    numeric_candidates = ["quantity", "unit_price", "total_ht", "total_ttc"]
    categorical_candidates = ["governorate", "city"]

    for col in numeric_candidates:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    for col in categorical_candidates:
        if col in work.columns:
            work[col] = work[col].fillna("Unknown").astype(str)

    keep_cols = [c for c in numeric_candidates + categorical_candidates if c in work.columns]
    if not keep_cols:
        raise HTTPException(
            status_code=400,
            detail=(
                "No usable columns found. Expected at least one of: "
                "quantity, unit_price, total_ht, total_ttc, governorate, city"
            ),
        )

    features = work[keep_cols].copy()
    categorical_cols = [c for c in categorical_candidates if c in features.columns]
    if categorical_cols:
        features = pd.get_dummies(features, columns=categorical_cols, dummy_na=False)

    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)

    return features


def _normalize_supplier_clusters(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["supplier_name", "cluster_id", "cluster_name"])

    supplier_col = "supplier_name" if "supplier_name" in df.columns else None
    cluster_col = "cluster" if "cluster" in df.columns else ("cluster_id" if "cluster_id" in df.columns else None)
    cluster_name_col = "cluster_name" if "cluster_name" in df.columns else None

    if supplier_col is None or cluster_col is None:
        return pd.DataFrame(columns=["supplier_name", "cluster_id", "cluster_name"])

    out = pd.DataFrame()
    out["supplier_name"] = df[supplier_col].astype(str)
    out["cluster_id"] = pd.to_numeric(df[cluster_col], errors="coerce").astype("Int64")
    out["cluster_name"] = df[cluster_name_col].astype(str) if cluster_name_col else None

    # Keep response deterministic and clean.
    out = out.sort_values(["cluster_id", "supplier_name"], na_position="last").reset_index(drop=True)
    return out


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


def _supplier_cluster_name_map(supplier_clusters: pd.DataFrame | None) -> dict[int, str]:
    if not isinstance(supplier_clusters, pd.DataFrame) or supplier_clusters.empty:
        return {}
    if "cluster" not in supplier_clusters.columns or "cluster_name" not in supplier_clusters.columns:
        return {}
    out: dict[int, str] = {}
    for _, row in supplier_clusters.iterrows():
        try:
            cluster_id = int(row.get("cluster"))
        except Exception:
            continue
        cluster_name = row.get("cluster_name")
        if cluster_name is None or str(cluster_name).strip().lower() == "nan":
            continue
        out.setdefault(cluster_id, str(cluster_name))
    return out


def _cluster_name_from_id(cluster_id: int) -> str:
    return f"Cluster {cluster_id + 1}"


def _align_supplier_feature_frame(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    aligned = features.copy()
    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = 0.0
    aligned = aligned.reindex(columns=feature_columns, fill_value=0.0).copy()
    for column in aligned.columns:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce").fillna(0.0)
    return aligned


def _supplier_prediction_confidence(model: Any, scaled_features: np.ndarray) -> float:
    if scaled_features is None or len(scaled_features) == 0:
        return float("nan")
    distances = model.transform(scaled_features)
    if distances is None or len(distances) == 0:
        return float("nan")
    min_distances = np.min(distances, axis=1)
    confidence = 1.0 / (1.0 + min_distances)
    return float(np.mean(confidence))


def _supplier_feature_summary(features: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for column in features.columns:
        series = pd.to_numeric(features[column], errors="coerce")
        summary[column] = {
            "mean": float(series.mean()) if not series.empty else 0.0,
            "std": float(series.std(ddof=0)) if not series.empty else 0.0,
        }
    return summary


def _supplier_distribution_shift(current_features: pd.DataFrame, training_summary: dict[str, Any] | None) -> float:
    if not isinstance(training_summary, dict) or not training_summary:
        return float("nan")

    shifts: list[float] = []
    for column in current_features.columns:
        current_series = pd.to_numeric(current_features[column], errors="coerce")
        if current_series.empty:
            continue
        baseline = training_summary.get(column)
        if not isinstance(baseline, dict):
            continue
        baseline_mean = float(baseline.get("mean", 0.0))
        current_mean = float(current_series.mean())
        reference = max(abs(baseline_mean), 1e-9)
        shifts.append(abs(current_mean - baseline_mean) / reference)

    if not shifts:
        return float("nan")
    return float(np.mean(shifts))


def _load_supplier_dataset_from_postgres() -> pd.DataFrame:
    ml_schema = os.getenv("ML_SCHEMA", "ml")
    dataset = load_ml_dataset_from_db(SUPPLIER_DB_DATASET_TABLE, schema_name=ml_schema)
    if not isinstance(dataset, pd.DataFrame) or dataset.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No supplier dataset found in PostgreSQL table {ml_schema}.{SUPPLIER_DB_DATASET_TABLE}. "
                "Populate the table before calling the supplier training endpoints."
            ),
        )
    return dataset


def _build_supplier_clusters_frame(dataset: pd.DataFrame, labels) -> pd.DataFrame:
    supplier_name_series = (
        dataset["supplier_name"].astype(str)
        if "supplier_name" in dataset.columns
        else pd.Series([f"Supplier {idx + 1}" for idx in range(len(dataset))])
    )
    clusters = pd.DataFrame(
        {
            "supplier_name": supplier_name_series.reset_index(drop=True),
            "cluster": pd.Series(labels).astype(int),
        }
    )
    if "supplier_id" in dataset.columns:
        clusters.insert(0, "supplier_id", dataset["supplier_id"].reset_index(drop=True))
    clusters["cluster_name"] = clusters["cluster"].map(_cluster_name_from_id)
    return clusters.sort_values(["cluster", "supplier_name"]).reset_index(drop=True)


def _train_supplier_artifact(dataset: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> tuple[dict[str, Any], pd.DataFrame, Any, pd.DataFrame]:
    work = dataset.copy().reset_index(drop=True)
    features = _prepare_supplier_features(work)
    rows = len(features)
    if rows < 2:
        raise HTTPException(status_code=400, detail="Supplier dataset must contain at least 2 usable rows.")
    if n_clusters < 2:
        raise HTTPException(status_code=400, detail="n_clusters must be >= 2")
    if rows < n_clusters:
        raise HTTPException(status_code=400, detail=f"n_clusters={n_clusters} exceeds the available rows ({rows}).")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)
    silhouette = float("nan")
    if len(np.unique(labels)) >= 2 and len(X_scaled) > len(np.unique(labels)):
        try:
            silhouette = float(silhouette_score(X_scaled, labels))
        except Exception:
            silhouette = float("nan")

    baseline_confidence = _supplier_prediction_confidence(model, X_scaled)
    supplier_clusters = _build_supplier_clusters_frame(work, labels)
    artifact = {
        "task": "supplier_train_api",
        "model": model,
        "scaler": scaler,
        "feature_columns": list(features.columns),
        "n_clusters": int(n_clusters),
        "supplier_clusters": supplier_clusters,
        "baseline_inertia": float(model.inertia_),
        "baseline_silhouette": silhouette,
        "baseline_confidence": baseline_confidence,
        "training_feature_summary": _supplier_feature_summary(features),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return artifact, features, X_scaled, supplier_clusters


def ensure_supplier_prediction_artifact() -> dict[str, Any]:
    artifact = load_artifact(SUPPLIER_MODEL_PATH)
    if isinstance(artifact, dict):
        return artifact

    dataset = _load_supplier_dataset_from_postgres()

    artifact, _, _, _ = _train_supplier_artifact(dataset, n_clusters=min(4, len(dataset)), random_state=42)
    save_versioned_artifact(artifact, SUPPLIER_MODEL_PATH)
    return artifact


def _build_supplier_features_from_request(payload: SupplierPredictRequest, feature_columns: list[str]) -> pd.DataFrame:
    row = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    df = _prepare_supplier_features(pd.DataFrame([row]))
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    df = df.reindex(columns=feature_columns, fill_value=0.0).copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _log_supplier_mlflow_run(
    *,
    run_name: str,
    X_scaled,
    y_pred,
    model,
    scaler,
    feature_columns: list[str],
    n_clusters: int,
    random_state: int,
    rows: int,
    comparison_label: str,
) -> None:
    if not MLFLOW_AVAILABLE:
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("supplier-clustering")

    labels = pd.Series(y_pred)
    sil = float("nan")
    if labels.nunique() >= 2 and len(labels) > labels.nunique():
        try:
            sil = float(silhouette_score(X_scaled, y_pred))
        except Exception:
            sil = float("nan")

    payload = {
        "task": "supplier_train_api",
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "n_clusters": int(n_clusters),
    }
    artifact_dir = Path("./mlartifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{run_name}.joblib"
    joblib.dump(payload, artifact_path)

    mlflow.set_experiment("supplier-clustering")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "n_clusters": int(n_clusters),
                "random_state": int(random_state),
                "rows": int(rows),
                "feature_count": int(len(feature_columns)),
                "comparison_label": comparison_label,
            }
        )
        mlflow.log_metrics(
            {
                "inertia": float(model.inertia_),
                "silhouette": sil,
            }
        )
        mlflow.log_artifact(str(artifact_path), artifact_path="artifacts")




@router.post("/lookup/supplier-clusters/save", response_model=SupplierClusterSaveResponse)
def save_supplier_clusters() -> SupplierClusterSaveResponse:
    ml_schema = os.getenv("ML_SCHEMA", "ml")
    result_table = os.getenv("ML_SUPPLIER_TABLE", "supplier_clusters")

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ml_schema or ""):
        raise HTTPException(status_code=400, detail=f"Invalid ML_SCHEMA={ml_schema!r}")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", result_table or ""):
        raise HTTPException(status_code=400, detail=f"Invalid ML_SUPPLIER_TABLE={result_table!r}")

    artifact = ensure_supplier_prediction_artifact()
    supplier_df = artifact.get("supplier_clusters") if isinstance(artifact, dict) else None
    if not isinstance(supplier_df, pd.DataFrame) or supplier_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "No supplier_clusters found in model artifact. "
                "Run supplier clustering notebook and save model first."
            ),
        )

    out = supplier_df.copy()
    if "cluster_id" in out.columns and "cluster" not in out.columns:
        out = out.rename(columns={"cluster_id": "cluster"})

    required_cols = ["supplier_name", "cluster", "cluster_name"]
    missing_cols = [c for c in required_cols if c not in out.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"supplier_clusters is missing required columns: {missing_cols}",
        )

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


@router.get("/train/supplier/dataset", response_model=SupplierPrepareDatasetResponse)
def get_supplier_prepared_dataset(limit: int = 20) -> SupplierPrepareDatasetResponse:
    n = max(1, min(limit, 5000))
    ds = _load_supplier_dataset_from_postgres()
    publish_data_metrics(SUPPLIER_MODEL_NAME, ds)

    return {
        "rows": int(len(ds)),
        "feature_columns": list(ds.columns),
        "preview": safe_head(ds, n=n),
    }


@router.post("/train/supplier/train", response_model=SupplierTrainResponse)
def train_supplier_model(n_clusters: int = 4, random_state: int = 42) -> SupplierTrainResponse:
    global _supplier_prepared_dataset, _supplier_train_results

    ds = _load_supplier_dataset_from_postgres()
    publish_data_metrics(SUPPLIER_MODEL_NAME, ds)
    record_retraining_trigger(SUPPLIER_MODEL_NAME, "manual_train_endpoint", rows=len(ds), n_clusters=n_clusters)

    _supplier_prepared_dataset = ds

    artifact, X, X_scaled, supplier_clusters = _train_supplier_artifact(ds, n_clusters=n_clusters, random_state=random_state)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_columns = artifact["feature_columns"]

    versioned_model_path = save_versioned_artifact(artifact, SUPPLIER_MODEL_PATH)
    if SUPPLIER_TRAIN_MODEL_PATH != SUPPLIER_MODEL_PATH:
        save_versioned_artifact(artifact, SUPPLIER_TRAIN_MODEL_PATH)

    publish_model_metrics(
        SUPPLIER_MODEL_NAME,
        {
            "inertia": float(model.inertia_),
            "silhouette": artifact.get("baseline_silhouette"),
            "confidence": artifact.get("baseline_confidence"),
            "quality_drop_ratio": 0.0,
        },
    )
    publish_drift_metrics(
        SUPPLIER_MODEL_NAME,
        {
            "distribution_shift": 0.0,
            "precision_drop_ratio": 0.0,
            "confidence_drop_ratio": 0.0,
            "detected": 0.0,
        },
    )

    comparison_k = 2 if n_clusters != 2 else 3
    comparison_enabled = len(X) >= comparison_k
    comparison_model = None
    comparison_labels = None
    if comparison_enabled:
        comparison_model = KMeans(n_clusters=comparison_k, random_state=random_state, n_init=10)
        comparison_labels = comparison_model.fit_predict(X_scaled)

    _log_supplier_mlflow_run(
        run_name=f"supplier_kmeans_k{n_clusters}",
        X_scaled=X_scaled,
        y_pred=supplier_clusters["cluster"].tolist(),
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        n_clusters=n_clusters,
        random_state=random_state,
        rows=len(X),
        comparison_label="selected",
    )
    if comparison_enabled and comparison_model is not None and comparison_labels is not None:
        _log_supplier_mlflow_run(
            run_name=f"supplier_kmeans_k{comparison_k}",
            X_scaled=X_scaled,
            y_pred=comparison_labels,
            model=comparison_model,
            scaler=scaler,
            feature_columns=feature_columns,
            n_clusters=comparison_k,
            random_state=random_state,
            rows=len(X),
            comparison_label="comparison",
        )

    label_counts = supplier_clusters["cluster"].value_counts().sort_index().to_dict()
    _supplier_train_results = {
        "model_path": SUPPLIER_MODEL_PATH,
        "versioned_model_path": versioned_model_path,
        "n_clusters": int(n_clusters),
        "rows": int(len(X)),
        "feature_columns": feature_columns,
        "inertia": float(model.inertia_),
        "cluster_counts": {str(k): int(v) for k, v in label_counts.items()},
    }

    log_event(
        logging.INFO,
        "supplier_training_completed",
        rows=int(len(X)),
        n_clusters=int(n_clusters),
        inertia=float(model.inertia_),
        silhouette=artifact.get("baseline_silhouette"),
        confidence=artifact.get("baseline_confidence"),
    )

    return {
        "model_path": SUPPLIER_MODEL_PATH,
        "versioned_model_path": versioned_model_path,
        "n_clusters": int(n_clusters),
        "rows": int(len(X)),
        "feature_columns": feature_columns,
        "inertia": float(model.inertia_),
    }


@router.post("/predict/supplier", response_model=SupplierPredictResponse)
def predict_supplier_cluster(payload: SupplierPredictRequest) -> SupplierPredictResponse:
    artifact = ensure_supplier_prediction_artifact()

    model = artifact.get("model")
    scaler = artifact.get("scaler")
    feature_columns = artifact.get("feature_columns")
    if model is None or scaler is None or not isinstance(feature_columns, list):
        raise HTTPException(status_code=500, detail="Supplier artifact is missing model/scaler/feature_columns.")

    input_df = _build_supplier_features_from_request(payload, feature_columns)
    scaled = scaler.transform(input_df)
    predicted_cluster_id = int(model.predict(scaled)[0])
    confidence = _supplier_prediction_confidence(model, scaled)
    set_model_metric(SUPPLIER_MODEL_NAME, "confidence", confidence)

    cluster_map = _supplier_cluster_name_map(artifact.get("supplier_clusters") if isinstance(artifact, dict) else None)
    predicted_cluster_name = cluster_map.get(predicted_cluster_id)

    note = None
    if not math.isnan(confidence) and confidence < 0.55:
        note = "Low confidence prediction. Consider retraining or checking data drift."
        log_event(
            logging.WARNING,
            "supplier_low_confidence_prediction",
            confidence=float(confidence),
            predicted_cluster_id=predicted_cluster_id,
        )

    return {
        "predicted_cluster_id": predicted_cluster_id,
        "predicted_cluster_name": predicted_cluster_name,
        "model_path": SUPPLIER_MODEL_PATH,
        "note": note,
    }


@router.get("/drift/supplier", response_model=SupplierDriftResponse)
def detect_supplier_drift() -> SupplierDriftResponse:
    artifact = ensure_supplier_prediction_artifact()
    model = artifact.get("model")
    scaler = artifact.get("scaler")
    feature_columns = artifact.get("feature_columns")
    if model is None or scaler is None or not isinstance(feature_columns, list) or not feature_columns:
        raise HTTPException(status_code=500, detail="Supplier artifact is missing model metadata.")

    ds = _load_supplier_dataset_from_postgres()
    publish_data_metrics(SUPPLIER_MODEL_NAME, ds)

    features = _prepare_supplier_features(ds)
    features = _align_supplier_feature_frame(features, feature_columns)
    scaled = scaler.transform(features)
    labels = model.predict(scaled)

    baseline_summary = artifact.get("training_feature_summary") if isinstance(artifact.get("training_feature_summary"), dict) else None
    distribution_shift = _supplier_distribution_shift(features, baseline_summary)

    baseline_silhouette = artifact.get("baseline_silhouette")
    current_silhouette = float("nan")
    if len(np.unique(labels)) >= 2 and len(scaled) > len(np.unique(labels)):
        try:
            current_silhouette = float(silhouette_score(scaled, labels))
        except Exception:
            current_silhouette = float("nan")

    baseline_confidence = artifact.get("baseline_confidence")
    current_confidence = _supplier_prediction_confidence(model, scaled)

    precision_drop_ratio = 0.0
    if isinstance(baseline_silhouette, (int, float)) and not math.isnan(float(baseline_silhouette)) and not math.isnan(current_silhouette):
        baseline_silhouette_value = float(baseline_silhouette)
        if baseline_silhouette_value != 0.0:
            precision_drop_ratio = max(0.0, (baseline_silhouette_value - current_silhouette) / abs(baseline_silhouette_value))

    confidence_drop_ratio = 0.0
    if isinstance(baseline_confidence, (int, float)) and not math.isnan(float(baseline_confidence)) and not math.isnan(current_confidence):
        baseline_confidence_value = float(baseline_confidence)
        if baseline_confidence_value != 0.0:
            confidence_drop_ratio = max(0.0, (baseline_confidence_value - current_confidence) / abs(baseline_confidence_value))

    drift_detected = bool(
        (not math.isnan(distribution_shift) and distribution_shift > 0.10)
        or precision_drop_ratio > 0.05
        or confidence_drop_ratio > 0.10
    )

    publish_model_metrics(SUPPLIER_MODEL_NAME, {"confidence": current_confidence})
    publish_drift_metrics(
        SUPPLIER_MODEL_NAME,
        {
            "distribution_shift": distribution_shift,
            "precision_drop_ratio": precision_drop_ratio,
            "confidence_drop_ratio": confidence_drop_ratio,
            "detected": 1.0 if drift_detected else 0.0,
        },
    )

    log_event(
        logging.WARNING if drift_detected else logging.INFO,
        "supplier_drift_check",
        drift_detected=drift_detected,
        distribution_shift=distribution_shift,
        precision_drop_ratio=precision_drop_ratio,
        confidence_drop_ratio=confidence_drop_ratio,
        baseline_silhouette=baseline_silhouette,
        current_silhouette=current_silhouette,
        baseline_confidence=baseline_confidence,
        current_confidence=current_confidence,
    )

    if drift_detected:
        log_event(
            logging.WARNING,
            "supplier_drift_detected",
            reason="threshold_exceeded",
            distribution_shift=distribution_shift,
            precision_drop_ratio=precision_drop_ratio,
            confidence_drop_ratio=confidence_drop_ratio,
        )

    return {
        "drift_detected": drift_detected,
        "distribution_shift": float(distribution_shift) if not math.isnan(distribution_shift) else 0.0,
        "precision_drop_ratio": float(precision_drop_ratio),
        "confidence_drop_ratio": float(confidence_drop_ratio),
        "baseline_silhouette": float(baseline_silhouette) if isinstance(baseline_silhouette, (int, float)) and not math.isnan(float(baseline_silhouette)) else None,
        "current_silhouette": float(current_silhouette) if not math.isnan(current_silhouette) else None,
        "baseline_confidence": float(baseline_confidence) if isinstance(baseline_confidence, (int, float)) and not math.isnan(float(baseline_confidence)) else None,
        "current_confidence": float(current_confidence) if not math.isnan(current_confidence) else None,
        "note": "Drift threshold exceeded." if drift_detected else "No drift detected.",
    }


