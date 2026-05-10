from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ml.database.pg_connect import DbConfig
from ml.database.repository import MlDataRepository

from .best_time_to_promote_service import BestTimeToPromoteService
from .best_time_to_sell_service import BestTimeToSellService
from .campaign_success_prediction_service import CampaignSuccessPredictionService
from .classification_concurrents_corrige1_service import CompetitorClassificationService
from .customer_churn_analytics_service import CustomerChurnAnalyticsService
from .product_matching_service import ProductMatchingService
from .supplier_classification_service import SupplierClassificationService


class MlServiceHub:
    def __init__(self, data_dir: Path | str, db_config: DbConfig | None = None) -> None:
        self.repository = MlDataRepository(data_dir=data_dir, db_config=db_config)
        self.competitors = CompetitorClassificationService(self.repository)
        self.campaigns = CampaignSuccessPredictionService(self.repository)
        self.sell = BestTimeToSellService(self.repository)
        self.promote = BestTimeToPromoteService(self.repository)
        self.suppliers = SupplierClassificationService(self.repository)
        self.churn = CustomerChurnAnalyticsService(self.repository)
        self.matcher = ProductMatchingService(self.repository)

    def overview(self) -> dict[str, object]:
        sales = self.repository.load_sales_frame()
        campaigns = self.repository.load_campaign_frame()
        suppliers = self.repository.load_supplier_frame()
        competitors = self.repository.load_competitor_frame()
        customers = self.repository.load_customer_behavior_frame()
        sougui_catalog = self.repository.load_sougui_catalog_frame()

        max_year = int(sales["year"].max()) if not sales.empty else datetime.now().year
        supported_years = sorted({max_year, max_year + 1, datetime.now().year, datetime.now().year + 1})

        supplier_geo = (
            suppliers[["governorate", "city"]]
            .drop_duplicates()
            .sort_values(["governorate", "city"])
        )
        cities_by_governorate: dict[str, list[str]] = {}
        for governorate, subset in supplier_geo.groupby("governorate"):
            cities_by_governorate[str(governorate)] = subset["city"].astype(str).tolist()

        workflows = [
            {
                "id": "competitorClassification",
                "title": "Competitive classification",
                "subtitle": "Classify a product against competitor patterns and category pressure.",
            },
            {
                "id": "campaignSuccess",
                "title": "Campaign success",
                "subtitle": "Estimate campaign performance before launch.",
            },
            {
                "id": "bestTimeToSell",
                "title": "Best time to sell",
                "subtitle": "Forecast the best future selling window from sales seasonality.",
            },
            {
                "id": "bestTimeToPromote",
                "title": "Best time to promote",
                "subtitle": "Recommend the strongest promotion window and discount range.",
            },
            {
                "id": "supplierClassification",
                "title": "Supplier classification",
                "subtitle": "Place a supplier into an operational segment from purchase history.",
            },
            {
                "id": "customerChurn",
                "title": "Customer churn intelligence",
                "subtitle": "Score retention risk, profile segments and surface at-risk customer groups.",
            },
        ]

        return {
            "status": "ready",
            "databaseStatus": self.repository.database_status(),
            "datasets": self.repository.source_catalog(),
            "metrics": [
                {"label": "ML workflows", "value": len(workflows), "hint": "production services"},
                {"label": "Sales signals", "value": len(sales), "hint": "rows in sales history"},
                {"label": "Campaign samples", "value": len(campaigns), "hint": "ad performance rows"},
                {"label": "Supplier profiles", "value": len(suppliers), "hint": "aggregated suppliers"},
                {"label": "Competitive products", "value": len(competitors), "hint": "products with market context"},
                {"label": "Customer events", "value": len(customers), "hint": "transactions used for churn scoring"},
                {"label": "Sougui catalog", "value": len(sougui_catalog), "hint": "products indexed for matching"},
            ],
            "options": {
                "channels": self.sell.available_channels(),
                "targetYears": supported_years,
                "governorates": sorted(cities_by_governorate.keys()),
                "citiesByGovernorate": cities_by_governorate,
                "souguiCategories": sorted(sougui_catalog["main_category"].dropna().astype(str).unique().tolist()),
                "promotionObjectives": [
                    {"value": "balanced", "label": "Balanced"},
                    {"value": "volume", "label": "Volume"},
                    {"value": "margin", "label": "Margin"},
                ],
            },
            "workflows": workflows,
        }
