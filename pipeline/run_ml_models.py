from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from config.db import DB_CONFIG, get_engine

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False


ROOT_DIR = Path(__file__).resolve().parents[1]
ML_DIR = ROOT_DIR / "ml_models"
OUTPUT_DIR = ML_DIR / "outputs"

# Load project .env once so child executions inherit available secrets.
load_dotenv(ROOT_DIR / ".env")


def _build_ml_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["OUTPUT_DIR"] = str(OUTPUT_DIR)
    env.setdefault("PYTHONPATH", str(ROOT_DIR))
    env.setdefault("PYTHONIOENCODING", "utf-8")

    # Ensure notebooks/scripts can connect without interactive password prompts.
    env.setdefault("PGHOST", str(DB_CONFIG.get("host", "localhost")))
    env.setdefault("PGPORT", str(DB_CONFIG.get("port", 5432)))
    env.setdefault("PGDATABASE", str(DB_CONFIG.get("database", "pi_bi")))
    env.setdefault("PGUSER", str(DB_CONFIG.get("user", "postgres")))
    env.setdefault("PGPASSWORD", str(DB_CONFIG.get("password", "")))
    env.setdefault("DB_PASSWORD", env.get("PGPASSWORD", ""))
    env.setdefault("DW_SCHEMA", "datawarehouse")
    return env


def _run_script(path: Path, timeout_sec: int = 900) -> Dict[str, Any]:
    env = _build_ml_env()

    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ML_DIR),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
        return {
            "file": path.name,
            "type": "python",
            "ok": proc.returncode == 0,
            "returncode": int(proc.returncode),
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
        }
    except Exception as e:
        return {
            "file": path.name,
            "type": "python",
            "ok": False,
            "error": str(e),
        }


def _run_notebook(path: Path, timeout_sec: int = 1800) -> Dict[str, Any]:
    env = _build_ml_env()
    executed_nb = OUTPUT_DIR / f"executed_{path.name}"

    old_env = os.environ.copy()
    os.environ.update(env)
    try:
        import papermill as pm

        pm.execute_notebook(
            input_path=str(path),
            output_path=str(executed_nb),
            cwd=str(ML_DIR),
            progress_bar=False,
            request_save_on_cell_execute=True,
            execution_timeout=timeout_sec,
            kernel_name="python3",
        )
        return {
            "file": path.name,
            "type": "notebook",
            "ok": True,
            "executed_notebook": str(executed_nb),
        }
    except Exception as e:
        return {
            "file": path.name,
            "type": "notebook",
            "ok": False,
            "error": str(e),
            "note": "Install papermill to execute notebooks automatically.",
        }
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def _discover_csv_outputs() -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for folder in [ML_DIR, OUTPUT_DIR]:
        if not folder.exists():
            continue
        for csv_file in sorted(folder.glob("*.csv")):
            out[csv_file.name] = csv_file
    return out


def _summarize_df(df: pd.DataFrame, max_rows: int = 10) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"rows": 0, "columns": [], "preview": []}
    return {
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "preview": df.head(max_rows).to_dict(orient="records"),
    }


def _prepare_churn_input_files() -> Dict[str, Any]:
    data_dir = ML_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "DimCustomer": "DimCustomer.csv",
        "dim_channel": "dim_channel.csv",
        "dim_category": "dim_category.csv",
        "DimProduct": "DimProduct.csv",
        "FactVentee": "FactVentee.csv",
        "FactShipping": "FactShipping.csv",
        "FactCampaign": "FactCampaign.csv",
    }

    report: Dict[str, Any] = {"ok": True, "files": []}
    try:
        engine = get_engine()
        for table_name, file_name in required.items():
            out_path = data_dir / file_name
            df = pd.read_sql(f'SELECT * FROM datawarehouse."{table_name}"', engine)
            df.to_csv(out_path, index=False)
            report["files"].append({"table": table_name, "path": str(out_path), "rows": int(df.shape[0])})
    except Exception as e:
        report["ok"] = False
        report["error"] = str(e)

    return report


def run_ml_models(tables: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
    """Execute ML files and read their outputs from files (CSV), not SQL tables."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Any] = {
        "meta": {
            "mode": "file_based",
            "ml_dir": str(ML_DIR),
            "output_dir": str(OUTPUT_DIR),
        }
    }

    run_report = []
    execute_ml = os.getenv("ML_EXECUTE_FILES", "1") in {"1", "true", "True"}

    outputs["churn_input_preparation"] = _prepare_churn_input_files()

    ml_files = [
        ML_DIR / "classification_concurrents.py",
        ML_DIR / "sougui_ml_step1.py",
        ML_DIR / "1_supplier_classification_postgres_v2.ipynb",
        ML_DIR / "2_best_time_to_sell_postgres_v2.ipynb",
        ML_DIR / "3_best_time_to_promote_postgres_v2.ipynb",
        ML_DIR / "customer_churn_analysis.ipynb",
    ]

    if execute_ml:
        py_timeout = int(os.getenv("ML_SCRIPT_TIMEOUT_SEC", "900"))
        nb_timeout = int(os.getenv("ML_NOTEBOOK_TIMEOUT_SEC", "1800"))

        for ml_file in ml_files:
            if not ml_file.exists():
                run_report.append({"file": ml_file.name, "ok": False, "error": "file not found"})
                continue

            if ml_file.suffix.lower() == ".py":
                run_report.append(_run_script(ml_file, timeout_sec=py_timeout))
            else:
                run_report.append(_run_notebook(ml_file, timeout_sec=nb_timeout))

    outputs["ml_file_runs"] = run_report

    csv_files = _discover_csv_outputs()
    outputs["ml_file_outputs"] = {
        "rows": len(csv_files),
        "files": list(csv_files.keys()),
    }

    def _read_csv(name: str) -> Dict[str, Any]:
        path = csv_files.get(name)
        if not path:
            return {"rows": 0, "note": f"Missing file {name}"}
        try:
            df = pd.read_csv(path)
            s = _summarize_df(df)
            s["path"] = str(path)
            return s
        except Exception as e:
            return {"rows": 0, "note": f"Failed to read {name}: {e}", "path": str(path)}

    def _read_csv_any(names: list[str]) -> Dict[str, Any]:
        for name in names:
            if name in csv_files:
                return _read_csv(name)
        return {"rows": 0, "note": f"Missing files: {', '.join(names)}"}

    outputs["supplier_clustering"] = _read_csv_any([
        "recommended_products_hybrid_clean.csv",
        "recommended_products_sougui_hybrid.csv",
    ])
    outputs["best_time_to_sell"] = _read_csv_any([
        "best_time_to_sell_monthly.csv",
        "best_price_fit_hybrid_clean.csv",
        "best_price_fit_products_sougui_hybrid.csv",
    ])
    outputs["best_time_to_promote"] = _read_csv_any([
        "best_time_to_promote_monthly.csv",
        "top_categories_hybrid_clean.csv",
        "top_categories_sougui_hybrid.csv",
    ])
    outputs["product_predictions"] = _read_csv_any(["customer_churn_predictions.csv"])
    outputs["high_risk_customers"] = _read_csv_any(["high_risk_customers.csv"])

    all_csv_outputs: Dict[str, Any] = {}
    for name, path in csv_files.items():
        try:
            df = pd.read_csv(path)
            s = _summarize_df(df, max_rows=5)
            s["path"] = str(path)
            all_csv_outputs[name] = s
        except Exception as e:
            all_csv_outputs[name] = {"rows": 0, "note": f"read error: {e}", "path": str(path)}
    outputs["all_csv_outputs"] = all_csv_outputs

    return outputs
