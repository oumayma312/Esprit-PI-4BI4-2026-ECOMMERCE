import os

from fastapi import FastAPI

from api_common import PROMOTE_MODEL_PATH, SELL_MODEL_PATH, SUPPLIER_MODEL_PATH
from promote_api import router as promote_router
from sell_api import router as sell_router
from supplier_api import router as supplier_router


app = FastAPI(title="Unified ML API", version="2.0.0")

app.include_router(supplier_router)
app.include_router(sell_router)
app.include_router(promote_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models_status() -> dict[str, dict[str, str | bool]]:
    return {
        "supplier": {"path": SUPPLIER_MODEL_PATH, "exists": os.path.exists(SUPPLIER_MODEL_PATH)},
        "sell": {"path": SELL_MODEL_PATH, "exists": os.path.exists(SELL_MODEL_PATH)},
        "promote": {"path": PROMOTE_MODEL_PATH, "exists": os.path.exists(PROMOTE_MODEL_PATH)},
    }