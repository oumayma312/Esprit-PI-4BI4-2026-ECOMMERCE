from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def sidecar_path(artifact_path: str | Path) -> Path:
    p = Path(artifact_path)
    return p.with_suffix(p.suffix + ".json")


def save_pt_artifact(obj: Any, artifact_path: str | Path, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    artifact = Path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": obj,
        "metadata": metadata or {},
    }

    serializer = "pickle"
    try:
        import torch  # type: ignore

        torch.save(payload, artifact)
        serializer = "torch"
    except Exception:
        with artifact.open("wb") as f:
            pickle.dump(payload, f)

    sidecar = sidecar_path(artifact)
    sidecar_data = {
        "artifact": str(artifact),
        "serializer": serializer,
        "metadata": _json_safe(metadata or {}),
    }
    sidecar.write_text(json.dumps(sidecar_data, indent=2, ensure_ascii=True), encoding="utf-8")

    return {
        "artifact_path": str(artifact),
        "sidecar_path": str(sidecar),
        "serializer": serializer,
    }


def load_artifact_metadata(artifact_path: str | Path) -> Dict[str, Any]:
    sidecar = sidecar_path(artifact_path)
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}
