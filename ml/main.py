import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import Counter

ERROR_COUNT = Counter("http_errors_total", "Total number of errors")

from api_common import PROMOTE_MODEL_PATH, SELL_MODEL_PATH, SUPPLIER_MODEL_PATH, artifact_display_path, artifact_exists
from campaign_api import CAMPAIGN_SUMMARY_PATH, router as campaign_router
from observability import log_event, record_error, setup_metrics
from promote_api import ensure_promote_prediction_artifact, router as promote_router
from sell_api import ensure_sell_prediction_artifact, router as sell_router
from sougui_api import router as sougui_router
from supplier_api import ensure_supplier_prediction_artifact, router as supplier_router
from ouma.api import router as ouma_router


app = FastAPI(title="Unified ML API", version="2.0.0")
SOUGUI_MODEL_PATH = Path(__file__).resolve().parent / "sougui_mlops" / "models" / "best_model.pkl"

app.include_router(supplier_router)
app.include_router(sell_router)
app.include_router(promote_router)
app.include_router(campaign_router)
app.include_router(sougui_router)
app.include_router(ouma_router)

setup_metrics(app)


@app.middleware("http")
async def count_errors(request, call_next):
    try:
        response = await call_next(request)

        if response.status_code >= 400:
            ERROR_COUNT.inc()

        return response

    except Exception:
        ERROR_COUNT.inc()
        raise
    
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    record_error("api", request.url.path, "http_exception", status_code=exc.status_code, detail=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    record_error("api", request.url.path, "validation_error", errors=exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log_event(40, "unhandled_exception", path=request.url.path, error_type=exc.__class__.__name__, error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


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
        "sougui": {"path": str(SOUGUI_MODEL_PATH), "exists": SOUGUI_MODEL_PATH.exists()},
         "ouma": {
            "path": str(Path("ouma/models/product_model.pkl")),
            "exists": os.path.exists("ouma/models/product_model.pkl"),
        },
    }
