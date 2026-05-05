from __future__ import annotations

import calendar
from datetime import date

import numpy as np
import pandas as pd


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalise_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    minimum = float(series.min())
    maximum = float(series.max())
    if np.isclose(maximum, minimum):
        return pd.Series(np.full(len(series), 0.5), index=series.index, dtype=float)
    return (series - minimum) / (maximum - minimum)


def confidence_from_signal(signal: float) -> float:
    bounded = 1 / (1 + np.exp(-signal))
    return float(clamp(bounded, 0.05, 0.99))


def find_window_date(target_year: int, month: int, weekday: int, occurrence: int = 2) -> str:
    month_dates = pd.date_range(date(target_year, month, 1), periods=calendar.monthrange(target_year, month)[1], freq="D")
    candidates = [item for item in month_dates if item.dayofweek == weekday]
    if not candidates:
        fallback = pd.Timestamp(date(target_year, month, 15))
        return fallback.date().isoformat()
    pick_index = min(max(occurrence - 1, 0), len(candidates) - 1)
    return candidates[pick_index].date().isoformat()


def month_name(month: int) -> str:
    return calendar.month_name[month]

