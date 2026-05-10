from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd

from ml.database.repository import MlDataRepository

from .exceptions import MlServiceError, MlValidationError
from .utils import clamp, confidence_from_signal, find_window_date, month_name, normalise_series


class BestTimeToPromoteService:
    def __init__(self, repository: MlDataRepository) -> None:
        self.repository = repository

    @cached_property
    def _sales_frame(self) -> pd.DataFrame:
        frame = self.repository.load_sales_frame()
        if frame.empty:
            raise MlServiceError("Sales snapshot is empty.")
        return frame

    @cached_property
    def _campaign_frame(self) -> pd.DataFrame:
        return self.repository.load_campaign_frame()

    def predict(self, target_year: int, channel: str, objective: str) -> dict[str, object]:
        channel_key = channel.strip() or "All channels"
        if objective not in {"balanced", "volume", "margin"}:
            raise MlValidationError("Promotion objective must be balanced, volume or margin.")

        sales = self._sales_frame
        if channel_key != "All channels":
            sales = sales[sales["channel_name"] == channel_key].copy()
            if sales.empty:
                raise MlValidationError(f"No sales history found for channel '{channel_key}'.")

        daily = (
            sales.groupby("sale_date", as_index=False)
            .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"), discount_value=("discount_value", "sum"))
            .sort_values("sale_date")
        )
        if len(daily) < 12:
            raise MlServiceError("Not enough sales history to recommend a promotion window.")

        daily["promo_flag"] = daily["discount_value"] > 0
        daily["discount_rate"] = np.where(
            daily["revenue"] + daily["discount_value"] > 0,
            daily["discount_value"] / (daily["revenue"] + daily["discount_value"]),
            0,
        )
        daily["month"] = daily["sale_date"].dt.month
        daily["weekday"] = daily["sale_date"].dt.dayofweek

        campaign_month_bonus = pd.Series(0.5, index=range(1, 13), dtype=float)
        campaigns = self._campaign_frame
        if not campaigns.empty:
            campaign_scores = campaigns.groupby("month")["views"].mean().reindex(range(1, 13), fill_value=0)
            campaign_month_bonus = normalise_series(campaign_scores).reindex(range(1, 13), fill_value=0.5)

        grouped = (
            daily.groupby(["month", "weekday"], as_index=False)
            .agg(
                avg_revenue=("revenue", "mean"),
                avg_quantity=("quantity", "mean"),
                promo_days=("promo_flag", "sum"),
                avg_discount_rate=("discount_rate", "mean"),
            )
        )

        grouped["revenue_score"] = normalise_series(grouped["avg_revenue"].fillna(0))
        grouped["quantity_score"] = normalise_series(grouped["avg_quantity"].fillna(0))
        grouped["promo_score"] = normalise_series(grouped["promo_days"].fillna(0))
        grouped["campaign_bonus"] = grouped["month"].map(campaign_month_bonus).fillna(0.5)
        grouped["efficiency_score"] = grouped["revenue_score"] - grouped["avg_discount_rate"].fillna(0)

        if objective == "volume":
            grouped["total_score"] = (
                0.35 * grouped["revenue_score"]
                + 0.25 * grouped["quantity_score"]
                + 0.2 * grouped["promo_score"]
                + 0.2 * grouped["campaign_bonus"]
            )
            multiplier = 1.15
        elif objective == "margin":
            grouped["total_score"] = (
                0.35 * grouped["revenue_score"]
                + 0.1 * grouped["quantity_score"]
                + 0.15 * grouped["promo_score"]
                + 0.15 * grouped["campaign_bonus"]
                + 0.25 * grouped["efficiency_score"]
            )
            multiplier = 0.85
        else:
            grouped["total_score"] = (
                0.4 * grouped["revenue_score"]
                + 0.2 * grouped["quantity_score"]
                + 0.15 * grouped["promo_score"]
                + 0.15 * grouped["campaign_bonus"]
                + 0.1 * grouped["efficiency_score"]
            )
            multiplier = 1.0

        top_row = grouped.sort_values("total_score", ascending=False).iloc[0]
        launch_date = find_window_date(target_year, int(top_row["month"]), int(top_row["weekday"]))
        recommended_discount = clamp(float(top_row["avg_discount_rate"]) * 100 * multiplier, 5, 30)
        if np.isnan(recommended_discount) or recommended_discount <= 0:
            recommended_discount = 10 if objective == "balanced" else 12 if objective == "volume" else 8

        monthly_baseline = daily.groupby("month")["revenue"].mean()
        baseline = float(monthly_baseline.get(int(top_row["month"]), daily["revenue"].mean()))
        uplift = ((float(top_row["avg_revenue"]) - baseline) / max(baseline, 1.0)) * 100
        uplift = round(float(clamp(uplift, -10, 80)), 1)

        signal = float(top_row["total_score"] * 4)
        confidence = confidence_from_signal(signal)
        status = "strong" if confidence >= 0.72 else "balanced" if confidence >= 0.5 else "watch"

        top_windows = grouped.sort_values("total_score", ascending=False).head(3)

        return {
            "workflow": "bestTimeToPromote",
            "headline": f"Launch the promotion around {launch_date}",
            "summary": "The promotion window blends sales momentum, historical discount behavior and campaign-friendly months.",
            "status": status,
            "confidence": round(confidence, 4),
            "metrics": [
                {"label": "Recommended discount", "value": f"{recommended_discount:.1f}%", "tone": "primary"},
                {"label": "Expected uplift", "value": f"{uplift:.1f}%", "tone": "success"},
                {"label": "Objective", "value": objective.title(), "tone": "neutral"},
            ],
            "insights": [
                f"Best month is {month_name(int(top_row['month']))}.",
                f"Recommended weekday is {pd.Timestamp(launch_date).day_name()}.",
                f"Historical promo intensity in that slot averages {float(top_row['avg_discount_rate']) * 100:.1f}%.",
            ],
            "result": {
                "targetYear": target_year,
                "channel": channel_key,
                "objective": objective,
                "launchDate": launch_date,
                "recommendedDiscountPercent": round(recommended_discount, 2),
                "expectedUpliftPercent": uplift,
                "topWindows": [
                    {
                        "month": month_name(int(row["month"])),
                        "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][int(row["weekday"])],
                        "score": round(float(row["total_score"]), 4),
                    }
                    for _, row in top_windows.iterrows()
                ],
            },
        }

