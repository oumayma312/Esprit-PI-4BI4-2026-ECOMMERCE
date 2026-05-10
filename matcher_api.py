from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from chatbot.config import AppConfig
from ml.database.pg_connect import DbConfig
from ml.database.repository import MlDataRepository
from ml.models.schemas import ProductMatcherCompareRequest
from ml.services.exceptions import MlServiceError, MlValidationError
from ml.services.product_matching_service import ProductMatchingService


config = AppConfig.from_env()
repository = MlDataRepository(data_dir=config.data_dir, db_config=DbConfig.from_env())
matcher = ProductMatchingService(repository)

app = FastAPI(title="Sougui Matcher API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.allowed_origins) or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sougui_assets_dir = config.root_dir / "sougui_photos"
if sougui_assets_dir.is_dir():
    app.mount("/assets/sougui", StaticFiles(directory=sougui_assets_dir), name="sougui-assets")


def translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, MlValidationError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@app.get("/overview")
def get_overview() -> dict[str, object]:
    try:
        return matcher.overview()
    except Exception as exc:  # pragma: no cover - standalone compatibility
        raise translate_error(exc) from exc


@app.get("/catalog")
def search_catalog(
    query: str = Query(default="", max_length=255),
    category: str = Query(default="", max_length=120),
    limit: int = Query(default=24, ge=1, le=60),
) -> dict[str, object]:
    try:
        return matcher.search(query=query, category=category, limit=limit)
    except Exception as exc:  # pragma: no cover - standalone compatibility
        raise translate_error(exc) from exc


@app.post("/compare")
def compare_products(payload: ProductMatcherCompareRequest) -> dict[str, object]:
    try:
        return matcher.compare(
            image_name=payload.image_name,
            product_name=payload.product_name,
            uploaded_image_data_url=payload.uploaded_image_data_url,
            uploaded_image_name=payload.uploaded_image_name,
            limit=payload.limit,
        )
    except Exception as exc:  # pragma: no cover - standalone compatibility
        raise translate_error(exc) from exc


@app.post("/match_live")
def match_live(payload: ProductMatcherCompareRequest) -> dict[str, object]:
    try:
        return matcher.compare(
            image_name=payload.image_name,
            product_name=payload.product_name,
            uploaded_image_data_url=payload.uploaded_image_data_url,
            uploaded_image_name=payload.uploaded_image_name,
            limit=payload.limit,
        )
    except Exception as exc:  # pragma: no cover - standalone compatibility
        raise translate_error(exc) from exc
