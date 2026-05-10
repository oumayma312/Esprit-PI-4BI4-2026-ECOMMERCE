from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ml.database.repository import MlDataRepository

from .exceptions import MlServiceError, MlValidationError
from .utils import confidence_from_signal, find_window_date, month_name


class BestTimeToSellService:
    def __init__(self, repository: MlDataRepository) -> None:
        self.repository = repository
        self._bundle_cache: dict[str, dict[str, object]] = {}

    @cached_property
    def _sales_frame(self) -> pd.DataFrame:
        frame = self.repository.load_sales_frame()
        if frame.empty:
            raise MlServiceError("Sales snapshot is empty.")
        return frame

    def available_channels(self) -> list[str]:
        channels = sorted(self._sales_frame["channel_name"].dropna().unique().tolist())
        return ["All channels", *[channel for channel in channels if channel != "All channels"]]

    def _calendar_features(self, dates: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "month": dates.dt.month,
                "weekday": dates.dt.dayofweek,
                "week_of_year": dates.dt.isocalendar().week.astype(int),
                "quarter": dates.dt.quarter,
                "day": dates.dt.day,
                "day_of_year": dates.dt.dayofyear,
                "is_month_start": dates.dt.is_month_start.astype(int),
                "is_month_end": dates.dt.is_month_end.astype(int),
            }
        )

    def _daily_frame(self, channel: str) -> pd.DataFrame:
        channel_key = channel.strip() or "All channels"
        if channel_key in self._bundle_cache:
            return self._bundle_cache[channel_key]["daily"]  # type: ignore[index]

        frame = self._sales_frame
        if channel_key != "All channels":
            frame = frame[frame["channel_name"] == channel_key].copy()
            if frame.empty:
                raise MlValidationError(f"No sales history found for channel '{channel_key}'.")

        daily = (
            frame.groupby("sale_date", as_index=False)
            .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"), discount_value=("discount_value", "sum"))
            .sort_values("sale_date")
        )
        if len(daily) < 20:
            raise MlServiceError("Not enough daily sales data to build a sell timing forecast.")

        model = RandomForestRegressor(n_estimators=260, max_depth=10, min_samples_leaf=2, random_state=42)
        feature_frame = self._calendar_features(daily["sale_date"])
        model.fit(feature_frame, daily["revenue"])

        bundle = {
            "daily": daily,
            "model": model,
            "average_revenue": float(daily["revenue"].mean()),
        }
        self._bundle_cache[channel_key] = bundle
        return daily

    def predict(self, target_year: int, channel: str) -> dict[str, object]:
        channel_key = channel.strip() or "All channels"
        self._daily_frame(channel_key)
        bundle = self._bundle_cache[channel_key]
        model: RandomForestRegressor = bundle["model"]  # type: ignore[assignment]
        average_revenue: float = bundle["average_revenue"]  # type: ignore[assignment]

        future_dates = pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31", freq="D")
        future_features = self._calendar_features(pd.Series(future_dates))
        predictions = model.predict(future_features)
        forecast = pd.DataFrame({"date": future_dates, "predictedRevenue": predictions})
        forecast["month"] = forecast["date"].dt.month
        forecast["weekday"] = forecast["date"].dt.dayofweek

        top_day = forecast.sort_values("predictedRevenue", ascending=False).iloc[0]
        monthly = (
            forecast.groupby("month", as_index=False)["predictedRevenue"]
            .mean()
            .sort_values("predictedRevenue", ascending=False)
        )
        top_windows = forecast.sort_values("predictedRevenue", ascending=False).head(3)

        signal = (float(top_day["predictedRevenue"]) - average_revenue) / max(average_revenue, 1.0)
        confidence = confidence_from_signal(signal)
        status = "strong" if confidence >= 0.72 else "balanced" if confidence >= 0.5 else "watch"

        return {
            "workflow": "bestTimeToSell",
            "headline": f"Best sell window: {pd.Timestamp(top_day['date']).strftime('%d %b %Y')}",
            "summary": "The forecast ranks future calendar windows using historical revenue seasonality from the sales warehouse.",
            "status": status,
            "confidence": round(confidence, 4),
            "metrics": [
                {"label": "Expected revenue", "value": f"{float(top_day['predictedRevenue']):.0f} TND", "tone": "primary"},
                {"label": "Best month", "value": month_name(int(monthly.iloc[0]['month'])), "tone": "success"},
                {"label": "Channel", "value": channel_key, "tone": "neutral"},
            ],
            "insights": [
                f"Predicted peak day is {pd.Timestamp(top_day['date']).day_name()} {pd.Timestamp(top_day['date']).strftime('%d %B %Y')}.",
                f"Average historical revenue baseline for this view is {average_revenue:.0f} TND.",
                f"Peak month recommendation is {month_name(int(monthly.iloc[0]['month']))}.",
            ],
            "result": {
                "targetYear": target_year,
                "channel": channel_key,
                "bestDate": pd.Timestamp(top_day["date"]).date().isoformat(),
                "bestMonth": month_name(int(monthly.iloc[0]["month"])),
                "bestWeekday": pd.Timestamp(top_day["date"]).day_name(),
                "expectedRevenue": round(float(top_day["predictedRevenue"]), 2),
                "topWindows": [
                    {
                        "date": pd.Timestamp(row["date"]).date().isoformat(),
                        "predictedRevenue": round(float(row["predictedRevenue"]), 2),
                    }
                    for _, row in top_windows.iterrows()
                ],
            },
        }

