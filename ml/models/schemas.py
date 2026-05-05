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

