import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
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
    load_dataset_from_sources,
    load_supplier_clusters_from_db,
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

router = APIRouter(tags=["supplier"])

SUPPLIER_TRAIN_MODEL_PATH = os.getenv("SUPPLIER_TRAIN_MODEL_PATH", "supplier_training_model_api.pkl")
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


def _build_supplier_features_from_request(payload: SupplierPredictRequest, feature_columns: list[str]) -> pd.DataFrame:
    row = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    df = pd.DataFrame([row])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_columns].copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0.0)
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

    artifact = load_artifact(SUPPLIER_MODEL_PATH)
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
    ds = load_dataset_from_sources(_supplier_prepared_dataset, SUPPLIER_DB_DATASET_TABLE, SUPPLIER_NOTEBOOK_DATASET_PATH)
    if ds is None or ds.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "No prepared supplier dataset found in API memory, PostgreSQL, or notebook snapshot. "
                "Expected PostgreSQL table ml.supplier or file at SUPPLIER_NOTEBOOK_DATASET_PATH "
                "(default: supplier_feature_dataset.parquet)."
            ),
        )

    return {
        "rows": int(len(ds)),
        "feature_columns": list(ds.columns),
        "preview": safe_head(ds, n=n),
    }


@router.post("/train/supplier/train", response_model=SupplierTrainResponse)
def train_supplier_model(n_clusters: int = 4, random_state: int = 42) -> SupplierTrainResponse:
    global _supplier_prepared_dataset, _supplier_train_results

    ds = load_dataset_from_sources(_supplier_prepared_dataset, SUPPLIER_DB_DATASET_TABLE, SUPPLIER_NOTEBOOK_DATASET_PATH)
    if ds is None or ds.empty:
        raise HTTPException(
            status_code=404,
            detail="No supplier dataset available. Check PostgreSQL table ml.supplier or the snapshot file.",
        )

    _supplier_prepared_dataset = ds

    if n_clusters < 2:
        raise HTTPException(status_code=400, detail="n_clusters must be >= 2")

    X = ds.copy()
    excluded_cols = {"supplier_id", "supplier_name", "cluster_kmeans", "cluster_dbscan", "cluster", "cluster_name", "note"}
    feature_cols = [c for c in X.columns if c not in excluded_cols]
    X = X[feature_cols].copy()
    X = X.select_dtypes(include=["number", "bool"]).copy()
    if X.empty:
        raise HTTPException(
            status_code=400,
            detail="No numeric supplier feature columns found in the dataset loaded from PostgreSQL.",
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)

    comparison_k = 2 if n_clusters != 2 else 3
    comparison_model = KMeans(n_clusters=comparison_k, random_state=random_state, n_init=10)
    comparison_labels = comparison_model.fit_predict(X_scaled)

    artifact = {
        "task": "supplier_train_api",
        "model": model,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "n_clusters": int(n_clusters),
    }
    versioned_model_path = save_versioned_artifact(artifact, SUPPLIER_TRAIN_MODEL_PATH)

    _log_supplier_mlflow_run(
        run_name=f"supplier_kmeans_k{n_clusters}",
        X_scaled=X_scaled,
        y_pred=labels,
        model=model,
        scaler=scaler,
        feature_columns=list(X.columns),
        n_clusters=n_clusters,
        random_state=random_state,
        rows=len(X),
        comparison_label="selected",
    )
    _log_supplier_mlflow_run(
        run_name=f"supplier_kmeans_k{comparison_k}",
        X_scaled=X_scaled,
        y_pred=comparison_labels,
        model=comparison_model,
        scaler=scaler,
        feature_columns=list(X.columns),
        n_clusters=comparison_k,
        random_state=random_state,
        rows=len(X),
        comparison_label="comparison",
    )

    label_counts = pd.Series(labels).value_counts().sort_index().to_dict()
    _supplier_train_results = {
        "model_path": SUPPLIER_TRAIN_MODEL_PATH,
        "versioned_model_path": versioned_model_path,
        "n_clusters": int(n_clusters),
        "rows": int(len(X)),
        "feature_columns": list(X.columns),
        "inertia": float(model.inertia_),
        "cluster_counts": {str(k): int(v) for k, v in label_counts.items()},
    }

    return {
        "model_path": SUPPLIER_TRAIN_MODEL_PATH,
        "versioned_model_path": versioned_model_path,
        "n_clusters": int(n_clusters),
        "rows": int(len(X)),
        "feature_columns": list(X.columns),
        "inertia": float(model.inertia_),
    }


@router.post("/predict/supplier", response_model=SupplierPredictResponse)
def predict_supplier_cluster(payload: SupplierPredictRequest) -> SupplierPredictResponse:
    artifact = load_artifact(SUPPLIER_MODEL_PATH)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Supplier model not found: {SUPPLIER_MODEL_PATH}")

    model = artifact.get("model")
    scaler = artifact.get("scaler")
    feature_columns = artifact.get("feature_columns")
    if model is None or scaler is None or not isinstance(feature_columns, list):
        raise HTTPException(status_code=500, detail="Supplier artifact is missing model/scaler/feature_columns.")

    input_df = _build_supplier_features_from_request(payload, feature_columns)
    scaled = scaler.transform(input_df)
    predicted_cluster_id = int(model.predict(scaled)[0])

    cluster_map = _supplier_cluster_name_map(artifact.get("supplier_clusters") if isinstance(artifact, dict) else None)
    predicted_cluster_name = cluster_map.get(predicted_cluster_id)

    return {
        "predicted_cluster_id": predicted_cluster_id,
        "predicted_cluster_name": predicted_cluster_name,
        "model_path": SUPPLIER_MODEL_PATH,
        "note": None,
    }


