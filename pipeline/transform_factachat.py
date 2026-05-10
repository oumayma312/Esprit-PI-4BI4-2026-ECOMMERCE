from __future__ import annotations

import numpy as np
import pandas as pd


def _to_number(series: pd.Series) -> pd.Series:
    """Parse numbers that might be formatted with ',' or '.' decimals and '%' signs."""

    if series is None:
        return series

    s = series.astype(object)
    s = s.where(~pd.isna(s), None)

    def norm_one(x):
        if x is None:
            return None
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        t = str(x).strip()
        if t == "" or t.lower() in {"(null)", "null", "nan", "none"}:
            return None
        t = t.replace("%", "").replace(" ", "")
        # locale-friendly: treat comma as decimal
        if "," in t and "." in t:
            # last separator is decimal
            if t.rfind(",") > t.rfind("."):
                t = t.replace(".", "")
                t = t.replace(",", ".")
            else:
                t = t.replace(",", "")
        else:
            t = t.replace(",", ".")
        try:
            return float(t)
        except Exception:
            return None

    out = s.map(norm_one)
    return pd.to_numeric(out, errors="coerce")


def transform_factachat(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the user's PowerQuery transformations to FactAchat."""

    if df is None or df.empty:
        return df

    out = df.copy()

    # quantity '.'->',' then number
    if "quantity" in out.columns:
        out["quantity"] = _to_number(out["quantity"])

    # price '.'->',' then number
    if "price" in out.columns:
        out["price"] = _to_number(out["price"])

    # remise type number (noop replacement in PQ)
    if "remise" in out.columns:
        out["remise"] = _to_number(out["remise"])

    # tva: remove '%' then numeric
    if "tva" in out.columns:
        out["tva"] = _to_number(out["tva"])

    # timbre numeric
    if "timbre" in out.columns:
        out["timbre"] = _to_number(out["timbre"])

    # totals: null -> 0 then numeric
    if "total_ht" in out.columns:
        th = out["total_ht"].where(~pd.isna(out["total_ht"]), 0)
        out["total_ht"] = _to_number(th)

    if "total_ttc" in out.columns:
        ttc = out["total_ttc"].where(~pd.isna(out["total_ttc"]), 0)
        out["total_ttc"] = _to_number(ttc)

    # Corrections
    if all(c in out.columns for c in ["total_ht", "total_ttc", "price", "quantity"]):
        price_qty = (out["price"].fillna(0) * out["quantity"].fillna(0))
        out["total_ht"] = out["total_ht"].fillna(0)
        out["total_ttc"] = out["total_ttc"].fillna(0)

        out.loc[out["total_ht"] == 0, "total_ht"] = np.where(
            price_qty[out["total_ht"] == 0] == 0,
            out.loc[out["total_ht"] == 0, "total_ttc"],
            price_qty[out["total_ht"] == 0],
        )

        out.loc[out["total_ttc"] == 0, "total_ttc"] = out.loc[out["total_ttc"] == 0, "total_ht"]

    # Reorder columns (if present)
    desired = [
        "FactAchatPK",
        "dateFK",
        "documentFK",
        "productFK",
        "supplierFK",
        "quantity",
        "price",
        "total_ht",
        "total_ttc",
        "remise",
        "tva",
        "timbre",
        "startDate",
        "endDate",
        "flag",
    ]
    present = [c for c in desired if c in out.columns]
    remaining = [c for c in out.columns if c not in present]
    out = out[present + remaining]

    return out
