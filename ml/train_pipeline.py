"""Training pipeline for churn prediction + customer segmentation.

Handles data loading (PostgreSQL primary, CSV fallback), feature derivation,
model training, evaluation, and MLflow tracking.

Produces at least 2 comparable MLflow runs (different algorithms).

Artifact structure:
    churn_artifacts/
    ├── models/          - Active model files
    ├── scalers/         - Active scaler files
    ├── config/          - Inference configuration
    └── versions/        - Timestamped versioned copies
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from api_common import (
    HTTPException,
    load_ml_dataset_from_db,
    save_versioned_artifact,
)
from churn_pipeline import APP_DIR, SEGMENT_LABELS


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = os.getenv("CHURN_ARTIFACTS_DIR", str(APP_DIR / "churn_artifacts"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_CHURN_EXPERIMENT", "churn-prediction")

# CSV fallback paths
CSV_DATA_PATHS = [
    APP_DIR / "data" / "FactVentee.csv",
    APP_DIR / "FactVentee.csv",
    Path("data/FactVentee.csv"),
    Path("FactVentee.csv"),
]

CSV_LABELS_PATHS = [
    APP_DIR / "customer_churn_predictions.csv",
    Path("customer_churn_predictions.csv"),
]

# Channel lookup
CHANNEL_CSV_PATHS = [
    APP_DIR / "data" / "dim_channel.csv",
    APP_DIR / "data" / "DimChannel.csv",
    Path("data/dim_channel.csv"),
]


def _ensure_mlflow() -> bool:
    """Ensure MLflow is available and configured with cross-platform artifact path."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()

        existing = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

        if existing is None:
            client.create_experiment(MLFLOW_EXPERIMENT_NAME)
        else:
            artifact_loc = existing.artifact_location or ""
            if artifact_loc.startswith("file:D:") or artifact_loc.startswith("file:C:"):
                client.delete_experiment(existing.experiment_id)
                client.create_experiment(MLFLOW_EXPERIMENT_NAME)

        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        return True
    except Exception as e:
        print(f"MLflow setup failed: {e}")
        return False


def _load_transactions_from_postgres() -> pd.DataFrame:
    """Load transaction data from PostgreSQL."""
    ml_schema = os.getenv("ML_SCHEMA", "data")
    table_name = os.getenv("CHURN_PG_TABLE", "FactVente")
    return load_ml_dataset_from_db(table_name, schema_name=ml_schema)


def _load_transactions_from_csv() -> pd.DataFrame:
    """Load transaction data from CSV files (fallback)."""
    for csv_path in CSV_DATA_PATHS:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                return df
    raise FileNotFoundError("No transaction CSV found")


def _load_label_data() -> pd.DataFrame:
    """Load pre-computed labels (churn_probability, segment) from CSV."""
    for csv_path in CSV_LABELS_PATHS:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                return df
    return pd.DataFrame()


def _load_channel_lookup() -> dict[str, str]:
    """Load channel name lookup from CSV."""
    lookup: dict[str, str] = {}
    for csv_path in CHANNEL_CSV_PATHS:
        if csv_path.exists():
            try:
                channel_df = pd.read_csv(csv_path)
                if "channelPK" in channel_df.columns and "channel" in channel_df.columns:
                    for _, row in channel_df[["channelPK", "channel"]].dropna().iterrows():
                        key = str(row["channelPK"]).strip()
                        value = str(row["channel"]).strip()
                        if key and value:
                            lookup[key] = value
                            try:
                                lookup[str(int(float(key)))] = value
                            except Exception:
                                pass
                    return lookup
            except Exception:
                continue
    return {}


def _derive_rfm_features(
    tx_df: pd.DataFrame,
    reference_date: pd.Timestamp,
    channel_lookup: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    """Derive RFM + derived features from raw transactions.

    Returns (customer_features_df, issues_list).
    """
    issues: list[str] = []
    df = tx_df.copy()

    required_cols = {"dateFK", "total_ttc", "quantity", "price", "channelFK", "productFK"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing transaction columns: {required_cols - set(df.columns)}")

    if "customerFK" not in df.columns:
        df["customerFK"] = "client_importe_1"
        issues.append("customerFK column missing: single customer assumed.")

    strict_dates = pd.to_datetime(df["dateFK"].astype(str), format="%Y%m%d", errors="coerce")
    loose_dates = pd.to_datetime(df["dateFK"], errors="coerce")
    df["dateFK"] = strict_dates.fillna(loose_dates)

    invalid = int(df["dateFK"].isna().sum())
    if invalid > 0:
        issues.append(f"{invalid} row(s) ignored due to invalid dateFK.")
        df = df.dropna(subset=["dateFK"]).copy()

    if df.empty:
        raise ValueError("No usable rows after date cleaning.")

    for col in ["total_ttc", "quantity", "price"]:
        before = df[col]
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        failures = int((before.notna() & pd.to_numeric(before, errors="coerce").isna()).sum())
        if failures > 0:
            issues.append(f"{col}: {failures} value(s) converted to 0.")

    if "FactVentePK" not in df.columns:
        df["FactVentePK"] = 1
        issues.append("FactVentePK missing: frequency calculated on row count.")

    df = df.sort_values(["customerFK", "dateFK"])

    customer_features = df.groupby("customerFK", as_index=False).agg(
        recency=("dateFK", lambda x: (reference_date - x.max()).days),
        first_purchase=("dateFK", "min"),
        last_purchase=("dateFK", "max"),
        frequency=("FactVentePK", "count"),
        monetary_total=("total_ttc", "sum"),
        monetary_mean=("total_ttc", "mean"),
        monetary_std=("total_ttc", "std"),
        quantity_total=("quantity", "sum"),
        quantity_mean=("quantity", "mean"),
        product_diversity=("productFK", "nunique"),
        channel_diversity=("channelFK", "nunique"),
        primary_channel=("channelFK", lambda x: x.mode().iloc[0] if not x.mode().empty else ""),
        avg_price=("price", "mean"),
        max_price=("price", "max"),
    )

    customer_features["monetary_std"] = customer_features["monetary_std"].fillna(0)
    customer_features["lifetime_days"] = (
        customer_features["last_purchase"] - customer_features["first_purchase"]
    ).dt.days
    customer_features["purchase_velocity"] = customer_features["frequency"] / (
        customer_features["lifetime_days"] / 30 + 1
    )

    # Compute last-3-tx monetary mean per customer (avoids deprecated .apply)
    last3 = df.sort_values("dateFK").groupby("customerFK").tail(3)
    last3_spend = last3.groupby("customerFK", as_index=False)["total_ttc"].mean().rename(
        columns={"total_ttc": "last3_monetary_mean"}
    )
    customer_features = customer_features.merge(last3_spend, on="customerFK", how="left")
    customer_features["monetary_trend"] = customer_features["last3_monetary_mean"] / (
        customer_features["monetary_mean"] + 1e-9
    )
    customer_features["recency_norm"] = customer_features["recency"] / (
        customer_features["lifetime_days"] + 1
    )

    return customer_features, issues


def _build_feature_matrix(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Build feature matrices for churn and segmentation models."""
    out = df.copy()
    issues: list[str] = []

    # Log-transform features
    for base_col in config.get("log_features", []):
        log_col = f"{base_col}_log"
        if log_col not in out.columns:
            if base_col in out.columns:
                vals = pd.to_numeric(out[base_col], errors="coerce").fillna(0)
            else:
                vals = pd.Series(0.0, index=out.index)
            out[log_col] = np.log1p(vals.clip(lower=0))

    # Channel dummy columns
    for col in config.get("churn_channel_dummy_columns", []):
        if col not in out.columns:
            out[col] = 0

    # Churn features
    churn_cols = config["churn_feature_columns"]
    missing_churn = [c for c in churn_cols if c not in out.columns]
    if missing_churn:
        issues.append(f"Missing churn columns: {missing_churn}")

    # Segment features
    seg_cols = config["segmentation_feature_columns"]
    missing_seg = [c for c in seg_cols if c not in out.columns]
    if missing_seg:
        issues.append(f"Missing segmentation columns: {missing_seg}")

    # Coerce to numeric
    all_cols = list(set(churn_cols + seg_cols))
    for col in all_cols:
        if col in out.columns:
            before = out[col]
            out[col] = pd.to_numeric(out[col], errors="coerce")
            failures = (out[col].isna() & before.notna()).sum()
            if failures > 0:
                issues.append(f"{col}: {int(failures)} non-numeric value(s)")

    out = out.fillna(0)
    return out, issues


def _generate_config(df: pd.DataFrame) -> dict[str, Any]:
    """Generate inference config from training data."""
    channels = df["primary_channel"].dropna().unique().tolist()
    channel_dummies = [f"ch_{ch}" for ch in channels if isinstance(ch, str) and ch.strip()]

    config = {
        "reference_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "global_floor": 90,
        "decision_threshold": 0.50,
        "decision_metric": "f1",
        "log_features": [
            "monetary_total",
            "quantity_total",
            "lifetime_days",
            "monetary_std",
            "purchase_velocity",
            "avg_price",
        ],
        "churn_feature_columns": [
            "frequency",
            "monetary_total_log",
            "monetary_trend",
            "product_diversity",
            "channel_diversity",
            "lifetime_days_log",
            "purchase_velocity_log",
            "avg_price_log",
        ] + channel_dummies,
        "churn_base_numeric_features": [
            "frequency",
            "monetary_total_log",
            "monetary_trend",
            "product_diversity",
            "channel_diversity",
            "lifetime_days_log",
            "purchase_velocity_log",
            "avg_price_log",
        ],
        "churn_channel_dummy_columns": channel_dummies,
        "segmentation_feature_columns": [
            "recency",
            "frequency",
            "monetary_total_log",
            "product_diversity",
            "purchase_velocity_log",
        ],
        "segment_count": 4,
    }
    return config


def _train_churn_model(
    X: pd.DataFrame,
    y: pd.Series,
    algorithm: str = "logistic_regression",
    random_state: int = 42,
    test_size: float = 0.2,
) -> tuple[Any, dict[str, float], str]:
    """Train a churn classifier and return (model, metrics, run_name)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if algorithm == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000, random_state=random_state, C=1.0, class_weight="balanced"
        )
    elif algorithm == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    run_name = f"churn_{algorithm}_rs{random_state}"
    return model, metrics, run_name


def _train_segmentation_model(
    X: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[Any, dict[str, float], str]:
    """Train a KMeans segmentation model and return (model, metrics, run_name)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)

    sil = float("nan")
    if len(set(labels)) >= 2 and len(labels) > len(set(labels)):
        try:
            sil = float(silhouette_score(X_scaled, labels))
        except Exception:
            pass

    metrics = {
        "silhouette": sil,
        "inertia": float(model.inertia_),
        "n_clusters": n_clusters,
    }

    run_name = f"segment_k{n_clusters}_rs{random_state}"
    return model, metrics, run_name


def _save_artifacts(
    churn_model: Any | None,
    churn_scaler: Any,
    segment_model: Any,
    segment_scaler: Any,
    config: dict[str, Any],
) -> dict[str, list[str]]:
    """Save trained artifacts in organized directory structure.

    Structure:
        churn_artifacts/
        ├── models/          (churn_model.joblib, segment_model.joblib)
        ├── scalers/         (churn_scaler.joblib, segment_scaler.joblib)
        ├── config/          (inference_config.json)
        └── versions/
            └── 20260428_120000/   (timestamped copy of all artifacts)

    Returns dict with keys: models, scalers, config, versions.
    """
    artifacts_path = Path(ARTIFACTS_DIR)
    models_dir = artifacts_path / "models"
    scalers_dir = artifacts_path / "scalers"
    config_dir = artifacts_path / "config"
    versions_dir = artifacts_path / "versions"

    for d in [models_dir, scalers_dir, config_dir, versions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version_dir = versions_dir / timestamp
    version_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, list[str]] = {
        "models": [],
        "scalers": [],
        "config": [],
        "versions": [],
    }

    # Save active churn model (only if not None - FIX for None model bug)
    if churn_model is not None:
        churn_model_path = models_dir / "churn_model.joblib"
        joblib.dump(churn_model, churn_model_path)
        saved["models"].append(str(churn_model_path))

        # Versioned copy
        versioned_model = version_dir / "churn_model.joblib"
        joblib.dump(churn_model, versioned_model)
        saved["versions"].append(str(versioned_model))

    # Save churn scaler
    churn_scaler_path = scalers_dir / "churn_scaler.joblib"
    joblib.dump(churn_scaler, churn_scaler_path)
    saved["scalers"].append(str(churn_scaler_path))

    versioned_churn_scaler = version_dir / "churn_scaler.joblib"
    joblib.dump(churn_scaler, versioned_churn_scaler)
    saved["versions"].append(str(versioned_churn_scaler))

    # Save segmentation model
    segment_model_path = models_dir / "segment_model.joblib"
    joblib.dump(segment_model, segment_model_path)
    saved["models"].append(str(segment_model_path))

    versioned_segment = version_dir / "segment_model.joblib"
    joblib.dump(segment_model, versioned_segment)
    saved["versions"].append(str(versioned_segment))

    # Save segment scaler
    segment_scaler_path = scalers_dir / "segment_scaler.joblib"
    joblib.dump(segment_scaler, segment_scaler_path)
    saved["scalers"].append(str(segment_scaler_path))

    versioned_seg_scaler = version_dir / "segment_scaler.joblib"
    joblib.dump(segment_scaler, versioned_seg_scaler)
    saved["versions"].append(str(versioned_seg_scaler))

    # Save config
    config_path = config_dir / "inference_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    saved["config"].append(str(config_path))

    versioned_config = version_dir / "config.json"
    with open(versioned_config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    saved["versions"].append(str(versioned_config))

    return saved


def train_churn_pipeline(
    algorithm: str = "logistic_regression",
    n_clusters: int = 4,
    random_state: int = 42,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run the full training pipeline.

    Returns dict with metrics, model paths, and config.
    """
    run_uuid = str(uuid.uuid4())[:8]
    reference_date = pd.Timestamp(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    channel_lookup = _load_channel_lookup()

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    try:
        tx_df = _load_transactions_from_postgres()
        data_source = "postgresql"
    except HTTPException:
        try:
            tx_df = _load_transactions_from_csv()
            data_source = "csv"
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No data source available: {e}. "
                    "Set PG* env vars or provide CSV files."
                ),
            )

    print(f"Loaded {len(tx_df)} transactions from {data_source}")

    # ------------------------------------------------------------------
    # Step 2: Derive RFM features
    # ------------------------------------------------------------------
    customer_df, derivation_issues = _derive_rfm_features(
        tx_df, reference_date, channel_lookup
    )
    print(f"Derived features for {len(customer_df)} customers")
    if derivation_issues:
        print(f"Derivation issues: {derivation_issues}")

    # ------------------------------------------------------------------
    # Step 3: Build feature matrix FIRST (creates log columns like monetary_total_log)
    # ------------------------------------------------------------------
    config = _generate_config(customer_df)
    prepared_df, build_issues = _build_feature_matrix(customer_df, config)
    if build_issues:
        print(f"Feature build issues: {build_issues}")

    # ------------------------------------------------------------------
    # Step 4: Load labels — merge into prepared_df (which has log columns)
    # ------------------------------------------------------------------
    labels_df = _load_label_data()
    has_labels = not labels_df.empty and "churn_probability" in labels_df.columns

    if has_labels and "customerFK" in prepared_df.columns and "customerFK" in labels_df.columns:
        prepared_df = prepared_df.merge(
            labels_df[["customerFK", "churn_probability", "segment"]],
            on="customerFK",
            how="left",
        )

    # ------------------------------------------------------------------
    # Step 5: Train churn model — use prepared_df (has log columns)
    # ------------------------------------------------------------------
    churn_model: Any | None = None
    churn_metrics: dict[str, float] = {}
    churn_run_name = ""
    churn_X: pd.DataFrame | None = None
    churn_y: pd.Series | None = None

    if has_labels:
        churn_mask = prepared_df["churn_probability"].notna()
        if churn_mask.sum() >= 10:
            churn_df = prepared_df[churn_mask].copy()
            churn_df["churn_label"] = (
                churn_df["churn_probability"] >= decision_threshold
            ).astype(int)

            if churn_df["churn_label"].nunique() >= 2:
                churn_X = churn_df[config["churn_feature_columns"]]
                churn_y = churn_df["churn_label"]

                churn_model, churn_metrics, churn_run_name = _train_churn_model(
                    churn_X,
                    churn_y,
                    algorithm=algorithm,
                    random_state=random_state,
                )
                print(f"Churn model trained: {churn_metrics}")
            else:
                churn_run_name = f"churn_{algorithm}_no_labels"
                print("Skipping churn training: insufficient label diversity")
        else:
            churn_run_name = f"churn_{algorithm}_insufficient_data"
            print("Skipping churn training: insufficient labeled data")
    else:
        churn_run_name = f"churn_{algorithm}_no_labels"
        print("Skipping churn training: no labels available")

    # ------------------------------------------------------------------
    # Step 6: Train segmentation model — use prepared_df (has log columns)
    # ------------------------------------------------------------------
    seg_X = prepared_df[config["segmentation_feature_columns"]]
    seg_X = seg_X.apply(pd.to_numeric, errors="coerce").fillna(0)

    segment_model, segment_metrics, segment_run_name = _train_segmentation_model(
        seg_X, n_clusters=n_clusters, random_state=random_state
    )

    segment_scaler = StandardScaler()
    segment_scaler.fit(seg_X)

    print(f"Segmentation model trained: {segment_metrics}")

    # ------------------------------------------------------------------
    # Step 7: Save artifacts in organized structure
    # ------------------------------------------------------------------
    saved = _save_artifacts(
        churn_model,  # FIX: now properly handles None
        StandardScaler().fit(churn_X) if churn_X is not None else StandardScaler(),
        segment_model,
        segment_scaler,
        config,
    )

    # ------------------------------------------------------------------
    # Step 8: Log to MLflow using mlflow.sklearn.log_model()
    # ------------------------------------------------------------------
    mlflow_available = _ensure_mlflow()

    if mlflow_available:
        # Churn model run
        if churn_model is not None:
            with mlflow.start_run(run_name=f"{churn_run_name}_{run_uuid}"):
                mlflow.log_params(
                    {
                        "algorithm": algorithm,
                        "random_state": random_state,
                        "decision_threshold": decision_threshold,
                        "n_customers": len(customer_df),
                        "n_transactions": len(tx_df),
                        "data_source": data_source,
                        "n_features": len(config["churn_feature_columns"]),
                    }
                )
                mlflow.log_metrics(churn_metrics)
                # FIX: Use mlflow.sklearn.log_model() for proper registry
                mlflow.sklearn.log_model(churn_model, artifact_path="churn_model")
                mlflow.sklearn.log_model(
                    StandardScaler().fit(churn_X), artifact_path="churn_scaler"
                )
                mlflow.log_artifact(str(saved["config"][0]), artifact_path="config")
                print(f"MLflow churn run logged: {churn_run_name}_{run_uuid}")

            # Comparison run with different algorithm
            comparison_algo = (
                "random_forest"
                if algorithm == "logistic_regression"
                else "logistic_regression"
            )
            try:
                comp_model, comp_metrics, comp_run_name = _train_churn_model(
                    churn_X,
                    churn_y,
                    algorithm=comparison_algo,
                    random_state=random_state,
                )

                with mlflow.start_run(run_name=f"{comp_run_name}_{run_uuid}"):
                    mlflow.log_params(
                        {
                            "algorithm": comparison_algo,
                            "random_state": random_state,
                            "decision_threshold": decision_threshold,
                            "n_customers": len(customer_df),
                            "n_transactions": len(tx_df),
                            "data_source": data_source,
                            "n_features": len(config["churn_feature_columns"]),
                            "comparison_of": churn_run_name,
                        }
                    )
                    mlflow.log_metrics(comp_metrics)
                    mlflow.sklearn.log_model(comp_model, artifact_path="churn_model")
                    mlflow.sklearn.log_model(
                        StandardScaler().fit(churn_X), artifact_path="churn_scaler"
                    )
                    mlflow.log_artifact(str(saved["config"][0]), artifact_path="config")
                    print(f"MLflow comparison run logged: {comp_run_name}_{run_uuid}")
            except Exception as e:
                print(f"Comparison run failed: {e}")

        # Segmentation run
        with mlflow.start_run(run_name=f"{segment_run_name}_{run_uuid}"):
            mlflow.log_params(
                {
                    "algorithm": "kmeans",
                    "n_clusters": n_clusters,
                    "random_state": random_state,
                    "n_customers": len(customer_df),
                    "n_transactions": len(tx_df),
                    "data_source": data_source,
                    "n_features": len(config["segmentation_feature_columns"]),
                }
            )
            mlflow.log_metrics(segment_metrics)
            # FIX: Use mlflow.sklearn.log_model() for proper registry
            mlflow.sklearn.log_model(segment_model, artifact_path="segment_model")
            mlflow.sklearn.log_model(segment_scaler, artifact_path="segment_scaler")
            mlflow.log_artifact(str(saved["config"][0]), artifact_path="config")
            print(f"MLflow segmentation run logged: {segment_run_name}_{run_uuid}")

    # ------------------------------------------------------------------
    # Return summary
    # ------------------------------------------------------------------
    saved_paths = [p for paths in saved.values() for p in paths]
    return {
        "run_uuid": run_uuid,
        "data_source": data_source,
        "n_customers": int(len(customer_df)),
        "n_transactions": int(len(tx_df)),
        "churn_model": churn_run_name,
        "churn_metrics": churn_metrics,
        "segment_model": segment_run_name,
        "segment_metrics": segment_metrics,
        "mlflow_available": mlflow_available,
        "saved_paths": saved_paths,
        "derivation_issues": derivation_issues,
    }
