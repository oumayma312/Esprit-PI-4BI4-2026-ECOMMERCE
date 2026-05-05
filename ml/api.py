from __future__ import annotations

from fastapi import APIRouter, HTTPException

from chatbot.config import AppConfig
from ml.database.pg_connect import DbConfig
from ml.models.schemas import (
    CampaignSuccessRequest,
    CompetitiveClassificationRequest,
    PromoteWindowRequest,
    SellWindowRequest,
    SupplierClassificationRequest,
)
from ml.services.exceptions import MlServiceError, MlValidationError
from ml.services.hub import MlServiceHub


def build_ml_router(config: AppConfig) -> APIRouter:
    router = APIRouter(prefix="/api/ml", tags=["ML"])
    hub = MlServiceHub(data_dir=config.data_dir, db_config=DbConfig.from_env())

    def translate_error(exc: Exception) -> HTTPException:
        if isinstance(exc, MlValidationError):
            return HTTPException(status_code=400, detail=str(exc))
        return HTTPException(status_code=500, detail=str(exc))

    @router.get("/overview")
    def get_overview() -> dict[str, object]:
        try:
            return hub.overview()
        except (MlServiceError, FileNotFoundError) as exc:
            raise translate_error(exc) from exc

    @router.post("/competitors/classify")
    def classify_competitor(payload: CompetitiveClassificationRequest) -> dict[str, object]:
        try:
            return hub.competitors.predict(
                product_name=payload.product_name,
                average_price=payload.average_price,
                competitor_count=payload.competitor_count,
            )
        except MlServiceError as exc:
            raise translate_error(exc) from exc

    @router.post("/campaigns/predict-success")
    def predict_campaign_success(payload: CampaignSuccessRequest) -> dict[str, object]:
        try:
            return hub.campaigns.predict(
                reach=payload.reach,
                impressions=payload.impressions,
                frequency=payload.frequency,
                budget_price=payload.budget_price,
            )
        except MlServiceError as exc:
            raise translate_error(exc) from exc

    @router.post("/sales/best-time")
    def predict_best_time_to_sell(payload: SellWindowRequest) -> dict[str, object]:
        try:
            return hub.sell.predict(target_year=payload.target_year, channel=payload.channel)
        except MlServiceError as exc:
            raise translate_error(exc) from exc

    @router.post("/promotions/best-time")
    def predict_best_time_to_promote(payload: PromoteWindowRequest) -> dict[str, object]:
        try:
            return hub.promote.predict(
                target_year=payload.target_year,
                channel=payload.channel,
                objective=payload.objective,
            )
        except MlServiceError as exc:
            raise translate_error(exc) from exc

    @router.post("/suppliers/classify")
    def classify_supplier(payload: SupplierClassificationRequest) -> dict[str, object]:
        try:
            return hub.suppliers.predict(
                governorate=payload.governorate,
                city=payload.city,
                total_quantity=payload.total_quantity,
                total_spend=payload.total_spend,
                purchase_count=payload.purchase_count,
                average_unit_price=payload.average_unit_price,
                active_products=payload.active_products,
            )
        except MlServiceError as exc:
            raise translate_error(exc) from exc

    return router

