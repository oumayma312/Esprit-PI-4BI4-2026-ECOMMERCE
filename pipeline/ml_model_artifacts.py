from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from pipeline.pt_artifact import load_artifact_metadata, save_pt_artifact
from pipeline.run_ml_models import run_ml_models

ROOT_DIR = Path(__file__).resolve().parents[1]
ML_DIR = ROOT_DIR / "ml_models"
ARTIFACT_DIR = ML_DIR / "artifacts"

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "classification_concurrents": {
        "kind": "script",
        "source": str(ML_DIR / "classification_concurrents.py"),
        "artifact": str(ARTIFACT_DIR / "classification_concurrents_model.pt"),
    },
    "sougui_step1": {
        "kind": "script",
        "source": str(ML_DIR / "sougui_ml_step1.py"),
        "artifact": str(ARTIFACT_DIR / "sougui_step1_model.pt"),
    },
    "supplier_classification_nb": {
        "kind": "notebook_proxy",
        "source": str(ML_DIR / "1_supplier_classification_postgres_v2.ipynb"),
        "artifact": str(ARTIFACT_DIR / "supplier_classification_nb_model.pt"),
        "ml_output_key": "supplier_clustering",
    },
    "best_time_to_sell_nb": {
        "kind": "notebook_proxy",
        "source": str(ML_DIR / "2_best_time_to_sell_postgres_v2.ipynb"),
        "artifact": str(ARTIFACT_DIR / "best_time_to_sell_nb_model.pt"),
        "ml_output_key": "best_time_to_sell",
    },
    "best_time_to_promote_nb": {
        "kind": "notebook_proxy",
        "source": str(ML_DIR / "3_best_time_to_promote_postgres_v2.ipynb"),
        "artifact": str(ARTIFACT_DIR / "best_time_to_promote_nb_model.pt"),
        "ml_output_key": "best_time_to_promote",
    },
    "customer_churn_nb": {
        "kind": "notebook_proxy",
        "source": str(ML_DIR / "customer_churn_analysis.ipynb"),
        "artifact": str(ARTIFACT_DIR / "customer_churn_nb_model.pt"),
        "ml_output_key": "product_predictions",
    },
}


def _run_script_export(spec: Dict[str, Any], timeout_sec: int = 1800) -> Dict[str, Any]:
    env = os.environ.copy()
    env["EXPORT_PT"] = "1"
    env["MODEL_ARTIFACT_PATH"] = spec["artifact"]
    env["OUTPUT_DIR"] = str(ARTIFACT_DIR)
    env.setdefault("PYTHONPATH", str(ROOT_DIR))
    env.setdefault("PYTHONIOENCODING", "utf-8")

    proc = subprocess.run(
        [sys.executable, spec["source"]],
        cwd=str(ML_DIR),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
    )

    ok = proc.returncode == 0 and Path(spec["artifact"]).exists()
    return {
        "ok": ok,
        "returncode": int(proc.returncode),
        "artifact_path": spec["artifact"],
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-30:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-30:]),
    }


def _export_notebook_proxy(spec: Dict[str, Any]) -> Dict[str, Any]:
    ml_out = run_ml_models()
    key = spec["ml_output_key"]
    section = ml_out.get(key) or {}

    metadata = {
        "kind": "notebook_proxy",
        "source": spec["source"],
        "ml_output_key": key,
        "measures": {
            "rows": section.get("rows", 0),
            "columns": section.get("columns", []),
            "path": section.get("path"),
        },
    }

    res = save_pt_artifact(obj={"proxy": key, "rows": section.get("rows", 0)}, artifact_path=spec["artifact"], metadata=metadata)
    return {
        "ok": True,
        "artifact_path": res["artifact_path"],
        "sidecar_path": res["sidecar_path"],
        "serializer": res["serializer"],
    }


def export_model_artifact(model_key: str) -> Dict[str, Any]:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {model_key}")

    spec = MODEL_REGISTRY[model_key]
    if spec["kind"] == "script":
        out = _run_script_export(spec)
    elif spec["kind"] == "notebook_proxy":
        out = _export_notebook_proxy(spec)
    else:
        raise ValueError(f"Unsupported model kind: {spec['kind']}")

    out["model_key"] = model_key
    out["kind"] = spec["kind"]
    return out


def export_all_model_artifacts() -> List[Dict[str, Any]]:
    results = []
    for key in MODEL_REGISTRY.keys():
        try:
            results.append(export_model_artifact(key))
        except Exception as e:
            results.append({
                "model_key": key,
                "ok": False,
                "error": str(e),
            })
    return results


def list_model_artifacts() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key, spec in MODEL_REGISTRY.items():
        artifact = Path(spec["artifact"])
        meta = load_artifact_metadata(artifact)
        out.append(
            {
                "model_key": key,
                "kind": spec["kind"],
                "source": spec["source"],
                "artifact_path": str(artifact),
                "exists": artifact.exists(),
                "metadata": meta.get("metadata", {}),
            }
        )
    return out


def get_model_measures(model_key: str) -> Dict[str, Any]:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {model_key}")

    artifact = MODEL_REGISTRY[model_key]["artifact"]
    meta = load_artifact_metadata(artifact)
    return (meta.get("metadata") or {}).get("measures", {})
