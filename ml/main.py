import os
from pathlib import Path

from fastapi import FastAPI

from api_common import PROMOTE_MODEL_PATH, SELL_MODEL_PATH, SUPPLIER_MODEL_PATH, artifact_display_path, artifact_exists
from campaign_api import CAMPAIGN_SUMMARY_PATH, router as campaign_router
from promote_api import ensure_promote_prediction_artifact, router as promote_router
from sell_api import ensure_sell_prediction_artifact, router as sell_router
from supplier_api import ensure_supplier_prediction_artifact, router as supplier_router


app = FastAPI(title="Unified ML API", version="2.0.0")

app.include_router(supplier_router)
app.include_router(sell_router)
app.include_router(promote_router)
app.include_router(campaign_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models_status() -> dict[str, dict[str, str | bool]]:
    ensure_supplier_prediction_artifact()
    ensure_sell_prediction_artifact()
    ensure_promote_prediction_artifact()
    return {
        "supplier": {"path": artifact_display_path(SUPPLIER_MODEL_PATH), "exists": artifact_exists(SUPPLIER_MODEL_PATH)},
        "sell": {"path": artifact_display_path(SELL_MODEL_PATH), "exists": artifact_exists(SELL_MODEL_PATH)},
        "promote": {"path": artifact_display_path(PROMOTE_MODEL_PATH), "exists": artifact_exists(PROMOTE_MODEL_PATH)},
        "campaign": {"path": str(Path(CAMPAIGN_SUMMARY_PATH)), "exists": os.path.exists(CAMPAIGN_SUMMARY_PATH)},
    }
