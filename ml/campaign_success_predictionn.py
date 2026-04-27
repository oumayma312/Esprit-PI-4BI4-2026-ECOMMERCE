from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psycopg
from campaign_business_rules import build_business_context, build_business_target, default_business_context
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pg_connect import DbConfig, get_connection, load_table, test_connection

ARTIFACTS_DIR = Path("mlops_artifacts")
MLFLOW_ARTIFACTS_DIR = Path("mlartifacts")
SUMMARY_PATH = ARTIFACTS_DIR / "campaign_summary.json"
DEFAULT_TABLE_CANDIDATES = ["fact_campaign", "FactCampaign"]
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
CAMPAIGN_EXPERIMENT_NAME = os.getenv("MLFLOW_CAMPAIGN_EXPERIMENT_NAME", "campaign-analytics")
CAMPAIGN_CLASSIFIER_MODEL_NAME = os.getenv("MLFLOW_CAMPAIGN_CLASSIFIER_MODEL_NAME", "campaign-success-classifier")
CAMPAIGN_REGRESSOR_MODEL_NAME = os.getenv("MLFLOW_CAMPAIGN_REGRESSOR_MODEL_NAME", "campaign-kpi-regressor")
CAMPAIGN_CLUSTERING_MODEL_NAME = os.getenv("MLFLOW_CAMPAIGN_CLUSTERING_MODEL_NAME", "campaign-segmentation-kmeans")

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


def safe_load_table(connection: psycopg.Connection, table_name: str, schema_name: str = "public") -> pd.DataFrame:
    try:
        return load_table(connection, table_name, schema_name=schema_name)
    except Exception as error:
        print(f"Chargement impossible pour {schema_name}.{table_name}: {error}")
        return pd.DataFrame()



def available_tables(connection: psycopg.Connection) -> pd.DataFrame:
    query = """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name;
    """
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=["table_schema", "table_name", "table_type"])



def find_best_table_name(connection: psycopg.Connection, candidates: list[str]) -> tuple[str, str] | None:
    tables_frame = available_tables(connection)
    if tables_frame.empty:
        return None

    tables_frame = tables_frame.assign(
        table_lower=tables_frame["table_name"].str.lower(),
        schema_lower=tables_frame["table_schema"].str.lower(),
    )

    for candidate in candidates:
        candidate_lower = candidate.lower()
        exact_match = tables_frame[tables_frame["table_lower"] == candidate_lower]
        if not exact_match.empty:
            row = exact_match.iloc[0]
            return row["table_schema"], row["table_name"]

    for _, row in tables_frame.iterrows():
        relation_name = row["table_name"].lower()
        if any(candidate.lower() in relation_name or relation_name in candidate.lower() for candidate in candidates):
            return row["table_schema"], row["table_name"]

    return None



def load_first_available_table(connection: psycopg.Connection, candidates: list[str]) -> tuple[pd.DataFrame, tuple[str, str] | None]:
    relation = find_best_table_name(connection, candidates)
    if relation is None:
        return pd.DataFrame(), None
    schema_name, table_name = relation
    return safe_load_table(connection, table_name, schema_name=schema_name), relation



def normalize_campaign_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if result.empty:
        return result

    if "date_pk" in result.columns:
        result["campaign_date"] = pd.to_datetime(result["date_pk"].astype(str), format="%Y%m%d", errors="coerce")

    numeric_like_columns = ["reach", "impressions", "frequency", "result", "views"]
    for column_name in numeric_like_columns:
        if column_name in result.columns:
            result[column_name] = pd.to_numeric(
                result[column_name].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    for column_name in ["price", "documentpk"]:
        if column_name in result.columns:
            result[column_name] = pd.to_numeric(
                result[column_name].astype(str).str.replace(r"[^0-9,.-]", "", regex=True).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    return result



def build_target_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, float | None, str, dict[str, Any] | None]:
    result = df.copy()
    if result.empty:
        return result, None, None, "empty_dataset", default_business_context()

    if "target" in result.columns:
        target_series = pd.to_numeric(result["target"], errors="coerce").fillna(0)
        unique_values = sorted(target_series.dropna().unique().tolist())
        target_series = (target_series > 0).astype(int) if len(unique_values) > 2 else target_series.astype(int)
        if target_series.nunique(dropna=True) >= 2:
            result["target"] = target_series
            return result, "target", None, "existing_target", build_business_context(result)

    campaign_columns = {"reach", "impressions", "frequency", "result", "views", "price"}
    if len(campaign_columns.intersection(result.columns)) >= 4:
        business_target, business_context, threshold, rule_name = build_business_target(result)
        if business_target.nunique(dropna=True) >= 2:
            result["target"] = business_target.astype(int)
            return result, "business_score", threshold, rule_name, business_context

    numeric_candidates = [
        "result",
        "views",
        "impressions",
        "reach",
        "clicks",
        "conversions",
        "sales",
        "revenue",
        "amount",
        "total_ttc",
        "total_ht",
        "price",
        "frequency",
        "ctr",
        "cvr",
        "roas",
        "engagement",
    ]
    for column_name in numeric_candidates:
        if column_name not in result.columns:
            continue
        numeric_series = pd.to_numeric(result[column_name], errors="coerce")
        if numeric_series.notna().sum() == 0:
            continue
        filled_series = numeric_series.fillna(numeric_series.median())
        for quantile, rule_prefix in ((0.75, "p75_threshold"), (0.50, "median_threshold")):
            threshold = float(filled_series.quantile(quantile))
            target_series = (filled_series >= threshold).astype(int)
            if target_series.nunique(dropna=True) >= 2:
                result["target"] = target_series
                return result, column_name, threshold, f"{rule_prefix}_{column_name}", build_business_context(result)

    numeric_frame = result.select_dtypes(include=np.number).copy()
    if not numeric_frame.empty:
        candidate_series = numeric_frame.iloc[:, 0]
        filled_candidate = candidate_series.fillna(candidate_series.median())
        threshold = float(filled_candidate.quantile(0.75))
        result["target"] = (filled_candidate >= threshold).astype(int)
        return result, candidate_series.name, threshold, f"fallback_numeric_{candidate_series.name}", build_business_context(result)

    result["target"] = 0
    return result, None, None, "fallback_constant_target", default_business_context()



def build_model_frame(df: pd.DataFrame, target_column: str, drop_columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    drop_columns = drop_columns or []
    modeling_frame = df.copy()
    if modeling_frame.empty:
        return pd.DataFrame(), pd.Series(dtype="float64")

    if "campaign_date" in modeling_frame.columns:
        base_date = modeling_frame["campaign_date"].min()
        modeling_frame["campaign_date_ordinal"] = (modeling_frame["campaign_date"] - base_date).dt.days

    columns_to_drop = set(drop_columns + [target_column, "campaign_date"])
    feature_frame = modeling_frame.drop(columns=[column for column in columns_to_drop if column in modeling_frame.columns])
    feature_frame = feature_frame.select_dtypes(include=[np.number]).copy()
    target_series = modeling_frame[target_column].copy()
    return feature_frame, target_series



def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "item"


def _sanitize_param_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return "none"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)


def _safe_numeric_metrics(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in payload.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            out[str(key)] = numeric
    return out


def _candidate_metrics_payload(candidate_records: list[dict[str, Any]] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not candidate_records:
        return out
    for record in candidate_records:
        model_name = _slugify(record.get("model", "candidate"))
        for key, value in record.items():
            if key == "model":
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                out[f"candidate_{model_name}_{key}"] = numeric
    return out


def _write_json_artifact(stem: str, payload: dict[str, Any]) -> Path:
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = MLFLOW_ARTIFACTS_DIR / f"{stem}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _mlflow_base_info() -> dict[str, Any]:
    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_name": CAMPAIGN_EXPERIMENT_NAME,
        "available": bool(MLFLOW_AVAILABLE),
    }


def _log_campaign_mlflow_run(
    *,
    run_name: str,
    task_name: str,
    model_name: str | None = None,
    model=None,
    feature_columns: list[str] | None = None,
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    candidate_records: list[dict[str, Any]] | None = None,
    tags: dict[str, Any] | None = None,
    register_model_name: str | None = None,
    extra_artifact_paths: list[str] | None = None,
) -> dict[str, Any]:
    info = _mlflow_base_info()
    info.update(
        {
            "run_name": run_name,
            "task_name": task_name,
            "model_name": model_name,
            "registered_model_name": None,
            "registered_model_version": None,
            "run_id": None,
            "model_uri": None,
            "error": None,
        }
    )
    if not MLFLOW_AVAILABLE:
        info["error"] = "mlflow_not_installed"
        return info

    artifact_paths = [path for path in (extra_artifact_paths or []) if path and Path(path).exists()]
    metadata_payload = {
        "task_name": task_name,
        "model_name": model_name,
        "feature_columns": feature_columns or [],
        "params": params or {},
        "metrics": _safe_numeric_metrics(metrics or {}),
        "candidate_records": candidate_records or [],
    }
    metadata_path = _write_json_artifact(f"{run_name}_summary", metadata_payload)
    artifact_paths.append(str(metadata_path))

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(CAMPAIGN_EXPERIMENT_NAME)
        client = MlflowClient() if MlflowClient is not None else None
        with mlflow.start_run(run_name=run_name) as run:
            info["run_id"] = run.info.run_id
            all_tags = {"task": task_name}
            if model_name:
                all_tags["model_name"] = model_name
            if tags:
                all_tags.update({str(key): str(value) for key, value in tags.items()})
            mlflow.set_tags(all_tags)

            log_params = {str(key): _sanitize_param_value(value) for key, value in (params or {}).items()}
            if feature_columns is not None:
                log_params["feature_count"] = int(len(feature_columns))
            if model_name:
                log_params["selected_model_name"] = model_name
            if log_params:
                mlflow.log_params(log_params)

            log_metrics = _safe_numeric_metrics(metrics or {})
            log_metrics.update(_candidate_metrics_payload(candidate_records))
            if log_metrics:
                mlflow.log_metrics(log_metrics)

            for artifact_path in artifact_paths:
                mlflow.log_artifact(str(artifact_path), artifact_path="artifacts")

            if model is not None:
                try:
                    mlflow.sklearn.log_model(model, artifact_path="model")
                    info["model_uri"] = f"runs:/{run.info.run_id}/model"
                except Exception as exc:
                    info["error"] = f"model_log_failed: {exc}"

            if register_model_name and info.get("model_uri") and client is not None:
                try:
                    try:
                        client.get_registered_model(register_model_name)
                    except Exception:
                        client.create_registered_model(register_model_name)
                except Exception:
                    pass
                try:
                    model_version = mlflow.register_model(model_uri=info["model_uri"], name=register_model_name)
                    info["registered_model_name"] = register_model_name
                    info["registered_model_version"] = str(getattr(model_version, "version", "") or "")
                except Exception as exc:
                    info["error"] = f"register_model_failed: {exc}"
    except Exception as exc:
        info["error"] = str(exc)

    return info


def export_mlops_artifact(task_name: str, model_name: str, model, feature_names: list[str], metrics: dict[str, float]) -> dict[str, str]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    slug = task_name.lower().replace(" ", "_")
    model_path = ARTIFACTS_DIR / f"{slug}_{timestamp}.joblib"
    metadata_path = ARTIFACTS_DIR / f"{slug}_{timestamp}.json"

    payload = {
        "task_name": task_name,
        "model_name": model_name,
        "feature_names": feature_names,
        "metrics": metrics,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "model": model,
    }
    joblib.dump(payload, model_path)

    metadata = {
        "task_name": task_name,
        "model_name": model_name,
        "feature_count": len(feature_names),
        "metrics": _safe_numeric_metrics(metrics),
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_path": str(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"model_path": str(model_path), "metadata_path": str(metadata_path)}



def train_classification(df: pd.DataFrame) -> dict[str, Any] | None:
    if df.empty or "target" not in df.columns:
        return None

    features, target = build_model_frame(df, "target", drop_columns=["date_pk", "documentpk"])
    if features.empty or target.nunique() < 2:
        return None

    x_train, x_test, y_train, y_test = train_test_split(features.fillna(0), target, test_size=0.25, random_state=42, stratify=target)
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample")),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    rows = []
    fitted_models = {}

    for model_name, model_pipeline in models.items():
        cv_results = cross_validate(model_pipeline, x_train, y_train, cv=cv, scoring=scoring, return_train_score=False)
        model_pipeline.fit(x_train, y_train)
        fitted_models[model_name] = model_pipeline
        y_pred = model_pipeline.predict(x_test)
        y_prob = model_pipeline.predict_proba(x_test)[:, 1]
        rows.append({
            "model": model_name,
            "accuracy_test": accuracy_score(y_test, y_pred),
            "precision_test": precision_score(y_test, y_pred, zero_division=0),
            "recall_test": recall_score(y_test, y_pred, zero_division=0),
            "f1_test": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc_test": roc_auc_score(y_test, y_prob),
            "accuracy_cv": cv_results["test_accuracy"].mean(),
            "precision_cv": cv_results["test_precision"].mean(),
            "recall_cv": cv_results["test_recall"].mean(),
            "f1_cv": cv_results["test_f1"].mean(),
            "roc_auc_cv": cv_results["test_roc_auc"].mean(),
        })

    results = pd.DataFrame(rows).sort_values("roc_auc_test", ascending=False)
    best_name = str(results.iloc[0]["model"])
    best_model = fitted_models[best_name]
    best_metrics = results.iloc[0].to_dict()
    export_paths = export_mlops_artifact(
        task_name="classification_success_campaign",
        model_name=best_name,
        model=best_model,
        feature_names=x_train.columns.tolist(),
        metrics={
            "accuracy_test": best_metrics.get("accuracy_test", np.nan),
            "f1_test": best_metrics.get("f1_test", np.nan),
            "roc_auc_test": best_metrics.get("roc_auc_test", np.nan),
        },
    )
    mlflow_info = _log_campaign_mlflow_run(
        run_name=f"campaign_classification_{_slugify(best_name)}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        task_name="campaign_classification",
        model_name=best_name,
        model=best_model,
        feature_columns=x_train.columns.tolist(),
        params={
            "rows_train": int(len(x_train)),
            "rows_test": int(len(x_test)),
            "cv_folds": 5,
            "candidate_count": len(models),
            "target_name": "target",
        },
        metrics={
            "selected_accuracy_test": best_metrics.get("accuracy_test"),
            "selected_precision_test": best_metrics.get("precision_test"),
            "selected_recall_test": best_metrics.get("recall_test"),
            "selected_f1_test": best_metrics.get("f1_test"),
            "selected_roc_auc_test": best_metrics.get("roc_auc_test"),
        },
        candidate_records=results.to_dict(orient="records"),
        tags={"component": "classification", "comparison_label": "selected"},
        register_model_name=CAMPAIGN_CLASSIFIER_MODEL_NAME,
        extra_artifact_paths=[export_paths["model_path"], export_paths["metadata_path"]],
    )

    return {
        "best_model": best_name,
        "metrics": best_metrics,
        "results": results.to_dict(orient="records"),
        "confusion_matrix": confusion_matrix(y_test, best_model.predict(x_test)).tolist(),
        "classification_report": classification_report(y_test, best_model.predict(x_test), zero_division=0),
        "artifact": export_paths,
        "mlflow": mlflow_info,
    }



def train_regression(df: pd.DataFrame) -> dict[str, Any] | None:
    if df.empty:
        return None

    regression_target_column = "views" if "views" in df.columns else ("impressions" if "impressions" in df.columns else None)
    if regression_target_column is None:
        return None

    features, target = build_model_frame(df, regression_target_column, drop_columns=["date_pk", "documentpk"])
    if features.empty or target.nunique() < 2:
        return None

    x_train, x_test, y_train, y_test = train_test_split(features.fillna(0), target, test_size=0.25, random_state=42)
    models = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    fitted_models = {}

    for model_name, model_pipeline in models.items():
        cv_results = cross_validate(model_pipeline, x_train, y_train, cv=cv, scoring={"mse": "neg_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}, return_train_score=False)
        model_pipeline.fit(x_train, y_train)
        fitted_models[model_name] = model_pipeline
        predictions = model_pipeline.predict(x_test)
        mse_value = mean_squared_error(y_test, predictions)
        rows.append({
            "model": model_name,
            "mse_test": mse_value,
            "rmse_test": float(np.sqrt(mse_value)),
            "mae_test": float(mean_absolute_error(y_test, predictions)),
            "r2_test": float(r2_score(y_test, predictions)),
            "mse_cv": float(-cv_results["test_mse"].mean()),
            "rmse_cv": float(np.sqrt(-cv_results["test_mse"].mean())),
            "mae_cv": float(-cv_results["test_mae"].mean()),
            "r2_cv": float(cv_results["test_r2"].mean()),
        })

    results = pd.DataFrame(rows).sort_values("rmse_test")
    best_name = str(results.iloc[0]["model"])
    best_model = fitted_models[best_name]
    best_metrics = results.iloc[0].to_dict()
    export_paths = export_mlops_artifact(
        task_name="regression_campaign_kpi",
        model_name=best_name,
        model=best_model,
        feature_names=x_train.columns.tolist(),
        metrics={
            "rmse_test": best_metrics.get("rmse_test", np.nan),
            "mae_test": best_metrics.get("mae_test", np.nan),
            "r2_test": best_metrics.get("r2_test", np.nan),
        },
    )
    mlflow_info = _log_campaign_mlflow_run(
        run_name=f"campaign_regression_{_slugify(best_name)}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        task_name="campaign_regression",
        model_name=best_name,
        model=best_model,
        feature_columns=x_train.columns.tolist(),
        params={
            "rows_train": int(len(x_train)),
            "rows_test": int(len(x_test)),
            "cv_folds": 5,
            "candidate_count": len(models),
            "target_name": regression_target_column,
        },
        metrics={
            "selected_rmse_test": best_metrics.get("rmse_test"),
            "selected_mae_test": best_metrics.get("mae_test"),
            "selected_r2_test": best_metrics.get("r2_test"),
        },
        candidate_records=results.to_dict(orient="records"),
        tags={"component": "regression", "comparison_label": "selected"},
        register_model_name=CAMPAIGN_REGRESSOR_MODEL_NAME,
        extra_artifact_paths=[export_paths["model_path"], export_paths["metadata_path"]],
    )

    return {
        "target_column": regression_target_column,
        "best_model": best_name,
        "metrics": best_metrics,
        "results": results.to_dict(orient="records"),
        "artifact": export_paths,
        "mlflow": mlflow_info,
    }



def cluster_campaigns(df: pd.DataFrame) -> dict[str, Any] | None:
    if df.empty:
        return None

    features, _ = build_model_frame(df, "target", drop_columns=["date_pk", "documentpk"])
    if features.empty:
        return None

    scaled = StandardScaler().fit_transform(features.fillna(0))
    k_values = range(2, 7)
    inertia = []
    silhouettes = []
    k_models = {}

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled)
        k_models[k] = kmeans
        inertia.append(float(kmeans.inertia_))
        silhouettes.append(float(silhouette_score(scaled, labels)) if len(np.unique(labels)) > 1 else np.nan)

    best_k = int(pd.Series(silhouettes, index=list(k_values)).idxmax())
    best_kmeans = k_models[best_k]
    kmeans_labels = best_kmeans.fit_predict(scaled)
    dbscan = DBSCAN(eps=1.2, min_samples=4)
    dbscan_labels = dbscan.fit_predict(scaled)
    pca = PCA(n_components=2, random_state=42).fit_transform(scaled)
    clustering_payload = {
        "best_k": best_k,
        "k_values": list(k_values),
        "elbow_inertia": inertia,
        "silhouette_curve": silhouettes,
        "dbscan_clusters": int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)),
        "dbscan_noise_count": int((dbscan_labels == -1).sum()),
    }
    clustering_artifact_path = _write_json_artifact(
        f"campaign_clustering_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        clustering_payload,
    )
    mlflow_info = _log_campaign_mlflow_run(
        run_name=f"campaign_clustering_k{best_k}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        task_name="campaign_clustering",
        model_name=f"kmeans_k{best_k}",
        model=best_kmeans,
        feature_columns=features.columns.tolist(),
        params={
            "rows": int(len(features)),
            "feature_count": int(features.shape[1]),
            "candidate_k_values": list(k_values),
            "dbscan_eps": 1.2,
            "dbscan_min_samples": 4,
        },
        metrics={
            "selected_best_k": best_k,
            "selected_kmeans_silhouette": float(silhouette_score(scaled, kmeans_labels)),
            "selected_kmeans_davies_bouldin": float(davies_bouldin_score(scaled, kmeans_labels)),
            "selected_dbscan_clusters": int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)),
            "selected_dbscan_noise_count": int((dbscan_labels == -1).sum()),
        },
        candidate_records=[
            {
                "model": f"kmeans_k{k}",
                "inertia": inertia[idx],
                "silhouette": silhouettes[idx],
            }
            for idx, k in enumerate(k_values)
        ],
        tags={"component": "clustering", "comparison_label": "selected"},
        register_model_name=CAMPAIGN_CLUSTERING_MODEL_NAME,
        extra_artifact_paths=[str(clustering_artifact_path)],
    )

    return {
        "best_k": best_k,
        "kmeans_silhouette": float(silhouette_score(scaled, kmeans_labels)),
        "kmeans_davies_bouldin": float(davies_bouldin_score(scaled, kmeans_labels)),
        "dbscan_clusters": int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)),
        "dbscan_noise_count": int((dbscan_labels == -1).sum()),
        "elbow_inertia": inertia,
        "silhouette_curve": silhouettes,
        "pca_preview": pca[:10].tolist(),
        "kmeans_labels_preview": kmeans_labels[:10].tolist(),
        "mlflow": mlflow_info,
    }



def dataframe_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "columns": [], "missing_columns": []}
    return {
        "rows": int(len(df)),
        "columns": [str(column) for column in df.columns],
        "missing_columns": [str(column) for column in df.columns if df[column].isna().any()],
        "head": df.head(5).to_dict(orient="records"),
    }



def build_summary(
    df_raw: pd.DataFrame,
    classification: dict[str, Any] | None,
    regression: dict[str, Any] | None,
    clustering: dict[str, Any] | None,
    target_rule: str,
    business_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mlflow_summary = _mlflow_base_info()
    mlflow_summary.update(
        {
            "classification": classification.get("mlflow") if isinstance(classification, dict) else None,
            "regression": regression.get("mlflow") if isinstance(regression, dict) else None,
            "clustering": clustering.get("mlflow") if isinstance(clustering, dict) else None,
            "pipeline": None,
        }
    )
    summary = {
        "dataset": dataframe_summary(df_raw),
        "target_rule": target_rule,
        "classification": classification,
        "regression": regression,
        "clustering": clustering,
        "business_context": business_context or default_business_context(),
        "mlflow": mlflow_summary,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary



def run_pipeline() -> dict[str, Any]:
    connection_ok = test_connection(DbConfig())
    if not connection_ok:
        return build_summary(pd.DataFrame(), None, None, None, "connection_failed", default_business_context())

    connection = get_connection()
    try:
        fact_campaign, fact_campaign_relation = load_first_available_table(connection, ["fact_campaign"])
        fact_campaign_alt, fact_campaign_alt_relation = load_first_available_table(connection, ["FactCampaign", "factcampaign"])
    finally:
        connection.close()

    frames = [frame for frame in [fact_campaign, fact_campaign_alt] if not frame.empty]
    if frames:
        df_raw = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates().reset_index(drop=True)
    else:
        df_raw = pd.DataFrame()

    if df_raw.empty:
        return build_summary(df_raw, None, None, None, "empty_dataset", default_business_context())

    df_raw.columns = [str(column).strip().lower() for column in df_raw.columns]
    df_raw = normalize_campaign_dataframe(df_raw)
    df_raw, target_source, target_threshold, target_rule, business_context = build_target_column(df_raw)
    classification = train_classification(df_raw)
    regression = train_regression(df_raw)
    clustering = cluster_campaigns(df_raw)

    summary = build_summary(df_raw, classification, regression, clustering, target_rule, business_context)
    summary["target_source"] = target_source
    summary["target_threshold"] = target_threshold
    summary["loaded_relations"] = [relation for relation in [fact_campaign_relation, fact_campaign_alt_relation] if relation is not None]
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pipeline_mlflow = _log_campaign_mlflow_run(
        run_name=f"campaign_pipeline_summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        task_name="campaign_pipeline_summary",
        model_name="campaign_summary",
        feature_columns=summary.get("dataset", {}).get("columns", []),
        params={
            "rows": summary.get("dataset", {}).get("rows", 0),
            "target_rule": target_rule,
            "target_source": target_source or "n/a",
            "loaded_relations": summary.get("loaded_relations", []),
        },
        metrics={
            "dataset_rows": summary.get("dataset", {}).get("rows", 0),
            "classification_available": int(isinstance(classification, dict)),
            "regression_available": int(isinstance(regression, dict)),
            "clustering_available": int(isinstance(clustering, dict)),
            "classification_roc_auc_test": classification.get("metrics", {}).get("roc_auc_test") if isinstance(classification, dict) else None,
            "regression_rmse_test": regression.get("metrics", {}).get("rmse_test") if isinstance(regression, dict) else None,
            "clustering_best_k": clustering.get("best_k") if isinstance(clustering, dict) else None,
        },
        tags={"component": "pipeline_summary"},
        extra_artifact_paths=[str(SUMMARY_PATH)],
    )
    if isinstance(summary.get("mlflow"), dict):
        summary["mlflow"]["pipeline"] = pipeline_mlflow
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary



def main() -> int:
    summary = run_pipeline()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Summary written to: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
