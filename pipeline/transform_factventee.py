from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlternateRowsSpec:
    offset: int
    keep: int
    skip: int


def _alternate_rows(df: pd.DataFrame, offset: int, keep: int, skip: int) -> pd.DataFrame:
    """Pandas version of PowerQuery Table.AlternateRows.

    Semantics (as used in PowerQuery): skip `offset` rows, then keep `keep` rows and
    skip `skip` rows repeatedly.

    Note: user requested to ignore sorting, so this operates on current row order.
    """

    if df is None or df.empty:
        return df
    offset = max(int(offset), 0)
    keep = max(int(keep), 0)
    skip = max(int(skip), 0)
    if keep == 0:
        return df.iloc[0:0].copy()

    tail = df.iloc[offset:].copy()
    period = keep + skip
    if period <= 0:
        return tail

    idx = np.arange(len(tail))
    mask = (idx % period) < keep
    return tail.loc[mask].copy()


def _split_to_list(value) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return [None]
    if isinstance(value, list):
        return value
    s = str(value)
    # PowerQuery splits on line feed "#(lf)".
    parts = s.split("\n")
    return parts if parts else [s]


def _to_number(series: pd.Series) -> pd.Series:
    """Parse numbers that might be formatted with ',' or '.' decimals."""

    if series is None:
        return series

    s = series.astype(object)
    # Keep None/NaN as NaN
    s = s.where(~pd.isna(s), None)

    def norm_one(x):
        if x is None:
            return None
        if isinstance(x, (int, float, np.integer, np.floating)):
            return x
        t = str(x).strip()
        if t == "" or t.lower() in {"(null)", "null", "nan"}:
            return None
        # remove percent sign if any
        t = t.replace("%", "")
        # remove spaces
        t = t.replace(" ", "")
        # If both separators exist, assume ',' thousands and '.' decimal (or vice versa)
        if "," in t and "." in t:
            # Heuristic: last separator is decimal
            if t.rfind(",") > t.rfind("."):
                t = t.replace(".", "")
                t = t.replace(",", ".")
            else:
                t = t.replace(",", "")
        else:
            # Single separator: treat comma as decimal
            t = t.replace(",", ".")
        try:
            return float(t)
        except Exception:
            return None

    out = s.map(norm_one)
    return pd.to_numeric(out, errors="coerce")


def transform_factventee(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the user's PowerQuery transformations to FactVentee.

    Notes:
      - Ignores sort steps (per user request).
      - Applies Table.AlternateRows filters exactly as provided.
      - Normalizes numeric columns to float where possible.
    """

    if df is None or df.empty:
        return df

    out = df.copy()

    # AlternateRows chain (ignore sorts)
    alternate_chain = [
        AlternateRowsSpec(1857, 483, 234),
        AlternateRowsSpec(1953, 19, 119),
        AlternateRowsSpec(1931, 2, 139),
        AlternateRowsSpec(1344, 13, 713),
    ]
    for spec in alternate_chain:
        out = _alternate_rows(out, spec.offset, spec.keep, spec.skip)

    # Split/expand price + quantity
    for col in ["price", "quantity"]:
        if col in out.columns:
            out[col] = out[col].map(_split_to_list)

    if "price" in out.columns:
        out = out.explode("price")
    if "quantity" in out.columns:
        out = out.explode("quantity")

    # quantity text replacements: "0" -> "1"
    if "quantity" in out.columns:
        q = out["quantity"].astype(object)
        q = q.where(~pd.isna(q), None)
        out["quantity"] = q.map(lambda x: "1" if str(x).strip() == "0" else x)

    # remise '.'->',' equivalent handled by _to_number
    # tva transformations
    if "tva" in out.columns:
        tva = out["tva"].astype(object).where(~pd.isna(out["tva"]), None)
        def fix_tva(x):
            if x is None:
                return None
            t = str(x).strip().replace("%", "")
            if t == "0,19":
                return "19"
            return t
        out["tva"] = tva.map(fix_tva)

    # total_ht special fill from total_ttc before numeric cast
    # Before that, ensure total_ht/total_ttc are treated as strings where needed.
    if "total_ht" in out.columns and "total_ttc" in out.columns:
        th = out["total_ht"].astype(object).where(~pd.isna(out["total_ht"]), None)
        ttc = out["total_ttc"].astype(object).where(~pd.isna(out["total_ttc"]), None)

        # AlternateRows(2612, 1, 721)
        out = _alternate_rows(out, 2612, 1, 721)

        def pick_total_ht(row):
            v = row.get("total_ht")
            if v is None:
                return row.get("total_ttc")
            s = str(v).strip()
            if s == "" or s in {"0", "(null)"}:
                return row.get("total_ttc")
            return v

        out["total_ht"] = out.apply(pick_total_ht, axis=1)

    # Numeric casts
    for col in ["price", "quantity", "remise", "tva", "timbre", "total_ht", "total_ttc"]:
        if col in out.columns:
            out[col] = _to_number(out[col])

    # Replace specific outliers with 0 (as in PowerQuery steps)
    replacements = [
        ("total_ht", 50000.0, 0.0),
        ("total_ht", 109504.5, 0.0),
        ("total_ttc", 109504.5, 0.0),
        ("total_ht", 4024.9, 0.0),
        ("total_ttc", 4024.9, 0.0),
        ("total_ht", 150.0, 0.0),
        ("total_ttc", 150.0, 0.0),
        ("total_ht", 40115.42, 0.0),
        ("total_ttc", 40115.42, 0.0),
        ("total_ht", 8373.0, 0.0),
        ("total_ttc", 8373.0, 0.0),
        ("total_ht", 1785.0, 0.0),
        ("total_ttc", 1785.0, 0.0),
        ("total_ht", 470.0, 0.0),
        ("total_ttc", 470.0, 0.0),
        ("total_ht", 1720.0, 0.0),
        ("total_ttc", 1720.0, 0.0),
        ("total_ht", 5460.0, 0.0),
        ("total_ttc", 5460.0, 0.0),
        ("total_ht", 224.34, 0.0),
        ("total_ttc", 224.34, 0.0),
        ("total_ht", 1871.0, 0.0),
        ("total_ttc", 1871.0, 0.0),
        ("total_ttc", 7775.0, 0.0),
        ("total_ht", 7775.0, 0.0),
        ("total_ttc", 5100.0, 0.0),
        ("total_ht", 5100.0, 0.0),
        ("total_ttc", 36667.5, 0.0),
        ("total_ht", 36667.5, 0.0),
        ("total_ht", 7470.0, 0.0),
        ("total_ttc", 7470.0, 0.0),
        ("total_ht", 605.0, 0.0),
        ("total_ttc", 605.0, 0.0),
        ("total_ttc", 50000.0, 0.0),
    ]
    for col, old, new in replacements:
        if col in out.columns:
            out.loc[out[col] == old, col] = new

    # Final fill: if total_ht is null or 0 -> total_ttc
    if "total_ht" in out.columns and "total_ttc" in out.columns:
        out["total_ht"] = out["total_ht"].where(~(out["total_ht"].isna() | (out["total_ht"] == 0)), out["total_ttc"])

    return out
