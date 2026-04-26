import os
import re
from glob import glob
from typing import Any

import joblib
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

SUPPLIER_MODEL_PATH = os.getenv("SUPPLIER_MODEL_PATH", "supplier_clustering_model.pkl")
SELL_MODEL_PATH = os.getenv("SELL_MODEL_PATH", "best_time_to_sell_model.pkl")
PROMOTE_MODEL_PATH = os.getenv("PROMOTE_MODEL_PATH", "best_time_to_promote_model.pkl")


def _load_local_env_file() -> None:
    candidates = [os.path.join(os.getcwd(), ".env"), os.path.join(os.path.dirname(__file__), ".env")]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception:
            continue


_load_local_env_file()


def _is_valid_identifier(name: str | None) -> bool:
    return bool(name) and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""))


def load_artifact(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        base, ext = os.path.splitext(path)
        candidates = sorted(glob(f"{base}_*{ext}"), reverse=True)
        if not candidates:
            return None
        path = candidates[0]
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError(f"Artifact at {path} must be a dict.")
    return artifact


def dated_artifact_path(path: str, timestamp: str | None = None) -> str:
    base, ext = os.path.splitext(path)
    stamp = timestamp or pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{stamp}{ext}"


def save_versioned_artifact(artifact: dict[str, Any], path: str) -> str:
    joblib.dump(artifact, path)
    versioned_path = dated_artifact_path(path)
    joblib.dump(artifact, versioned_path)
    return versioned_path


def load_dataframe_snapshot(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()

    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".pkl") or lower.endswith(".pickle"):
        obj = pd.read_pickle(path)
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, dict):
            for key in ["dataset", "features", "feat_df", "train_df"]:
                if key in obj and isinstance(obj[key], pd.DataFrame):
                    return obj[key]
        return pd.DataFrame()
    if lower.endswith(".csv"):
        return pd.read_csv(path)

    return pd.DataFrame()


def strip_export_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    metadata_cols = {
        "exported_at_utc",
        "saved_at_utc",
        "run_id",
        "created_at",
    }
    drop_cols = [c for c in df.columns if c in metadata_cols]
    if not drop_cols:
        return df
    return df.drop(columns=drop_cols).copy()


def load_dataset_from_sources(
    prepared_dataset: pd.DataFrame | None,
    table_name: str,
    snapshot_path: str,
    schema_name: str | None = None,
) -> pd.DataFrame:
    ds = pd.DataFrame()

    try:
        ds = load_ml_dataset_from_db(table_name, schema_name=schema_name)
    except HTTPException as exc:
        if exc.status_code != 503:
            raise

    if ds is None or ds.empty:
        if isinstance(prepared_dataset, pd.DataFrame) and not prepared_dataset.empty:
            ds = prepared_dataset.copy()

    if ds is None or ds.empty:
        ds = load_dataframe_snapshot(snapshot_path)

    return strip_export_metadata_columns(ds)


def load_ml_dataset_from_db(table_name: str, schema_name: str | None = None) -> pd.DataFrame:
    db_host = os.getenv("PGHOST") or os.getenv("DB_HOST")
    db_port = os.getenv("PGPORT") or os.getenv("DB_PORT") or "5432"
    db_name = os.getenv("PGDATABASE") or os.getenv("DB_NAME")
    db_user = os.getenv("PGUSER") or os.getenv("DB_USER")
    db_password = os.getenv("PGPASSWORD") or os.getenv("DB_PASSWORD")
    ml_schema = schema_name or os.getenv("ML_SCHEMA", "ml")

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
            detail=(
                "PostgreSQL configuration is missing in the API process. "
                f"Missing: {', '.join(missing)}. "
                "Set them as environment variables or add a .env file next to main.py/api_common.py."
            ),
        )

    if not _is_valid_identifier(ml_schema) or not _is_valid_identifier(table_name):
        raise HTTPException(status_code=400, detail="Invalid schema or table name for PostgreSQL dataset lookup.")

    try:
        engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
            pool_pre_ping=True,
        )

        query = text(f'SELECT * FROM "{ml_schema}"."{table_name}"')
        with engine.begin() as conn:
            return pd.read_sql(query, conn)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load PostgreSQL dataset from {ml_schema}.{table_name}: {exc}",
        ) from exc


def _native_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, pd.Timedelta):
        return str(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def safe_head(df: Any, n: int = 20) -> list[dict[str, Any]]:
    if isinstance(df, pd.DataFrame):
        out = df.head(n).copy()
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].astype(str)
        records = out.to_dict(orient="records")
        return [{k: _native_value(v) for k, v in record.items()} for record in records]
    return []


def ensure_datetime_series(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ds" in out.columns:
        out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
        out = out[out["ds"].notna()].copy()
    return out


def forecast_from_artifact_model(artifact: dict[str, Any], horizon_days: int = 730) -> pd.DataFrame:
    model = artifact.get("model")
    if model is None:
        return pd.DataFrame(columns=["ds", "yhat"])

    if hasattr(model, "get_forecast"):
        fc = model.get_forecast(steps=horizon_days)
        yhat = fc.predicted_mean
        out = pd.DataFrame({"ds": yhat.index, "yhat": yhat.values})
        return ensure_datetime_series(out)

    if hasattr(model, "predict") and hasattr(model, "history"):
        hist = getattr(model, "history", None)
        if hist is None or "ds" not in hist.columns:
            return pd.DataFrame(columns=["ds", "yhat"])
        last_ds = pd.to_datetime(hist["ds"], errors="coerce").max()
        if pd.isna(last_ds):
            return pd.DataFrame(columns=["ds", "yhat"])
        future = pd.DataFrame({"ds": pd.date_range(start=last_ds + pd.Timedelta(days=1), periods=horizon_days, freq="D")})
        pred = model.predict(future)
        cols = [c for c in ["ds", "yhat", "yhat_lower", "yhat_upper"] if c in pred.columns]
        out = pred[cols].copy()
        return ensure_datetime_series(out)

    return pd.DataFrame(columns=["ds", "yhat"])


def compute_sell_recommendation_from_forecast(forecast_df: pd.DataFrame) -> dict[str, Any]:
    if forecast_df.empty or "yhat" not in forecast_df.columns:
        return {"best_month": None, "best_date": None, "best_value": None}
    fc = ensure_datetime_series(forecast_df)
    if fc.empty:
        return {"best_month": None, "best_date": None, "best_value": None}
    fc = fc.sort_values("ds").copy()
    fc["month"] = fc["ds"].dt.to_period("M").astype(str)
    monthly = fc.groupby("month")["yhat"].sum()
    best_month = str(monthly.idxmax()) if len(monthly) else None
    i = fc["yhat"].idxmax()
    best_row = fc.loc[i]
    return {
        "best_month": best_month,
        "best_date": str(pd.to_datetime(best_row["ds"]).date()),
        "best_value": float(best_row["yhat"]),
    }


def compute_promote_recommendation_from_forecast(forecast_df: pd.DataFrame) -> dict[str, Any]:
    if forecast_df.empty or "yhat" not in forecast_df.columns:
        return {"best_month": None, "best_day": None, "best_day_value": None, "monthly_2027": []}
    fc = ensure_datetime_series(forecast_df)
    if fc.empty:
        return {"best_month": None, "best_day": None, "best_day_value": None, "monthly_2027": []}
    fc = fc.sort_values("ds").copy()
    fc_2027 = fc[fc["ds"].dt.year == 2027].copy()
    if not fc_2027.empty:
        fc = fc_2027
    fc["month"] = fc["ds"].dt.to_period("M").astype(str)
    monthly = fc.groupby("month")["yhat"].sum().reset_index(name="predicted_ca_sum")
    best_month = str(monthly.loc[monthly["predicted_ca_sum"].idxmin(), "month"]) if len(monthly) else None
    i = fc["yhat"].idxmin()
    best_row = fc.loc[i]
    return {
        "best_month": best_month,
        "best_day": str(pd.to_datetime(best_row["ds"]).date()),
        "best_day_value": float(best_row["yhat"]),
        "monthly_2027": monthly.to_dict(orient="records"),
    }


def compute_sell_monthly_from_forecast(forecast_df: pd.DataFrame) -> pd.DataFrame:
    fc = ensure_datetime_series(forecast_df)
    if fc.empty or "yhat" not in fc.columns:
        return pd.DataFrame(columns=["month", "expected_revenue_sum", "is_best_month"])

    fc = fc.sort_values("ds").copy()
    fc_2027 = fc[fc["ds"].dt.year == 2027].copy()
    if not fc_2027.empty:
        fc = fc_2027

    monthly = (
        fc.assign(month=fc["ds"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)["yhat"]
        .sum()
        .rename(columns={"yhat": "expected_revenue_sum"})
    )
    if monthly.empty:
        monthly["is_best_month"] = []
        return monthly

    best_month = monthly.loc[monthly["expected_revenue_sum"].idxmax(), "month"]
    monthly["is_best_month"] = monthly["month"].eq(best_month)
    return monthly


def compute_promote_monthly_from_forecast(forecast_df: pd.DataFrame) -> pd.DataFrame:
    fc = ensure_datetime_series(forecast_df)
    if fc.empty or "yhat" not in fc.columns:
        return pd.DataFrame(columns=["month", "predicted_ca_sum", "promotion_pct", "promotion_display", "is_best_month"])

    fc = fc.sort_values("ds").copy()
    fc_2027 = fc[fc["ds"].dt.year == 2027].copy()
    if not fc_2027.empty:
        fc = fc_2027

    monthly = (
        fc.assign(month=fc["ds"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)["yhat"]
        .sum()
        .rename(columns={"yhat": "predicted_ca_sum"})
    )
    if monthly.empty:
        monthly["promotion_pct"] = []
        monthly["promotion_display"] = []
        monthly["is_best_month"] = []
        return monthly

    s = monthly["predicted_ca_sum"].astype(float)
    s_min = float(s.min())
    s_max = float(s.max())
    promo_low = 0.20
    promo_high = 0.50
    if abs(s_max - s_min) < 1e-9:
        strength = pd.Series(0.5, index=monthly.index)
    else:
        strength = 1.0 - (s - s_min) / (s_max - s_min)
    monthly["promotion_pct"] = promo_low + strength * (promo_high - promo_low)
    monthly["promotion_display"] = (monthly["promotion_pct"] * 100).round(1).astype(str) + "%"

    best_month = monthly.loc[monthly["predicted_ca_sum"].idxmin(), "month"]
    monthly["is_best_month"] = monthly["month"].eq(best_month)
    return monthly


def load_supplier_clusters_from_db() -> pd.DataFrame:
    db_host = os.getenv("PGHOST") or os.getenv("DB_HOST")
    db_port = os.getenv("PGPORT") or os.getenv("DB_PORT") or "5432"
    db_name = os.getenv("PGDATABASE") or os.getenv("DB_NAME")
    db_user = os.getenv("PGUSER") or os.getenv("DB_USER")
    db_password = os.getenv("PGPASSWORD") or os.getenv("DB_PASSWORD")
    ml_schema = os.getenv("ML_SCHEMA", "ml")
    table_name = os.getenv("ML_SUPPLIER_TABLE", "supplier_clusters")

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
            detail=(
                "PostgreSQL configuration is missing in the API process. "
                f"Missing: {', '.join(missing)}. "
                "Set them as environment variables or add a .env file next to main.py/api_common.py."
            ),
        )

    try:
        engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
            pool_pre_ping=True,
        )
        with engine.begin() as conn:
            cols_q = text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema_name
                  AND table_name = :table_name
                """
            )
            table_cols = {
                str(r[0]).lower()
                for r in conn.execute(cols_q, {"schema_name": ml_schema, "table_name": table_name}).fetchall()
            }

            if "run_id" in table_cols and "created_at" in table_cols:
                query = text(
                    f'''SELECT *
                        FROM "{ml_schema}"."{table_name}"
                        WHERE run_id = (
                            SELECT run_id
                            FROM "{ml_schema}"."{table_name}"
                            ORDER BY created_at DESC NULLS LAST, run_id DESC
                            LIMIT 1
                        )
                        ORDER BY cluster, supplier_name'''
                )
            else:
                query = text(f'SELECT * FROM "{ml_schema}"."{table_name}" ORDER BY cluster, supplier_name')

            return pd.read_sql(query, conn)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load PostgreSQL supplier clusters from {ml_schema}.{table_name}: {exc}",
        ) from exc
