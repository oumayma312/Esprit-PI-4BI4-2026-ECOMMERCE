from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MlBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class CompetitiveClassificationRequest(MlBaseModel):
    product_name: str = Field(alias="productName", min_length=2, max_length=120)
    average_price: float = Field(alias="averagePrice", ge=0, le=100_000)
    competitor_count: int = Field(alias="competitorCount", default=1, ge=0, le=100)


class CampaignSuccessRequest(MlBaseModel):
    reach: int = Field(ge=0, le=10_000_000)
    impressions: int = Field(ge=0, le=20_000_000)
    frequency: float = Field(ge=0, le=20)
    budget_price: float = Field(alias="budgetPrice", ge=0, le=1_000_000)


class SellWindowRequest(MlBaseModel):
    target_year: int = Field(alias="targetYear", ge=2024, le=2035)
    channel: str = Field(default="All channels", min_length=1, max_length=80)


class PromoteWindowRequest(MlBaseModel):
    target_year: int = Field(alias="targetYear", ge=2024, le=2035)
    channel: str = Field(default="All channels", min_length=1, max_length=80)
    objective: Literal["balanced", "volume", "margin"] = "balanced"


class SupplierClassificationRequest(MlBaseModel):
    governorate: str = Field(min_length=2, max_length=80)
    city: str = Field(min_length=2, max_length=80)
    total_quantity: float = Field(alias="totalQuantity", ge=0, le=1_000_000)
    total_spend: float = Field(alias="totalSpend", ge=0, le=100_000_000)
    purchase_count: int = Field(alias="purchaseCount", ge=0, le=100_000)
    average_unit_price: float = Field(alias="averageUnitPrice", ge=0, le=1_000_000)
    active_products: int = Field(alias="activeProducts", default=1, ge=0, le=100_000)


class CustomerChurnScenarioRequest(MlBaseModel):
    channel: str = Field(default="All channels", min_length=1, max_length=80)
    recency: float = Field(ge=0, le=3650)
    frequency: float = Field(ge=1, le=100_000)
    monetary_total: float = Field(alias="monetaryTotal", ge=0, le=100_000_000)
    monetary_mean: float = Field(alias="monetaryMean", ge=0, le=10_000_000)
    quantity_total: float = Field(alias="quantityTotal", ge=0, le=10_000_000)
    product_diversity: float = Field(alias="productDiversity", ge=1, le=100_000)
    avg_price: float = Field(alias="avgPrice", ge=0, le=10_000_000)
    customer_lifetime_days: float = Field(alias="customerLifetimeDays", ge=0, le=10_000)
    monetary_std: float = Field(alias="monetaryStd", default=0, ge=0, le=10_000_000)


class ProductMatcherCompareRequest(MlBaseModel):
    image_name: str | None = Field(alias="imageName", default=None, max_length=255)
    product_name: str | None = Field(alias="productName", default=None, max_length=255)
    uploaded_image_data_url: str | None = Field(alias="uploadedImageDataUrl", default=None)
    uploaded_image_name: str | None = Field(alias="uploadedImageName", default=None, max_length=255)
    limit: int = Field(default=6, ge=1, le=12)
