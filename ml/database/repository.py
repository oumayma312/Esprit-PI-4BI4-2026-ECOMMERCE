from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import psycopg

from .pg_connect import DbConfig, get_connection, load_table


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    return result


def _pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {str(column).strip().lower(): str(column).strip().lower() for column in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _currency_to_float(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9,\.\-]", "", regex=True)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_date_key(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    valid_index = numeric[numeric > 0].index
    if len(valid_index) == 0:
        return result
    rendered = numeric.loc[valid_index].astype(int).astype(str).str.zfill(8)
    result.loc[valid_index] = pd.to_datetime(rendered, format="%Y%m%d", errors="coerce")
    return result


def _clean_label(series: pd.Series, fallback: str = "Unknown") -> pd.Series:
    cleaned = series.fillna(fallback).astype(str).str.strip()
    cleaned = cleaned.replace({"": fallback, "nan": fallback, "None": fallback})
    return cleaned


class MlDataRepository:
    def __init__(self, data_dir: Path | str, db_config: DbConfig | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.db_config = db_config or DbConfig.from_env()
        self._frame_cache: dict[str, pd.DataFrame] = {}
        self._source_cache: dict[str, str] = {}
        self._database_checked = False
        self._database_connected = False
        self._database_message = "Database has not been checked yet."

    def source_catalog(self) -> dict[str, str]:
        return dict(self._source_cache)

    def database_status(self) -> dict[str, object]:
        self._ensure_database_status()
        return {
            "enabled": self.db_config.enabled,
            "connected": self._database_connected,
            "message": self._database_message,
            "schema": self.db_config.schema,
        }

    def _ensure_database_status(self) -> None:
        if self._database_checked:
            return

        self._database_checked = True
        if not self.db_config.enabled:
            self._database_connected = False
            self._database_message = "Database usage disabled. Falling back to CSV snapshots."
            return

        connection = None
        try:
            connection = get_connection(self.db_config)
            self._database_connected = True
            self._database_message = "PostgreSQL connection ready."
        except psycopg.Error as exc:
            self._database_connected = False
            self._database_message = f"PostgreSQL unavailable. Using CSV snapshots instead: {exc}"
        finally:
            if connection is not None:
                connection.close()

    def _find_csv_path(self, candidates: Iterable[str]) -> Path | None:
        existing = {path.name.lower(): path for path in self.data_dir.iterdir() if path.is_file()}
        for candidate in candidates:
            match = existing.get(candidate.lower())
            if match is not None:
                return match
        return None

    def _load_table(
        self,
        cache_key: str,
        db_candidates: list[str],
        csv_candidates: list[str],
    ) -> pd.DataFrame:
        if cache_key in self._frame_cache:
            return self._frame_cache[cache_key].copy()

        frame: pd.DataFrame | None = None
        source = "unknown"

        self._ensure_database_status()
        if self._database_connected:
            connection = None
            try:
                connection = get_connection(self.db_config)
                for table_name in db_candidates:
                    try:
                        frame = load_table(connection, table_name, self.db_config.schema)
                        source = f"postgresql:{self.db_config.schema}.{table_name}"
                        break
                    except psycopg.Error:
                        continue
            except psycopg.Error as exc:
                self._database_connected = False
                self._database_message = f"PostgreSQL unavailable. Using CSV snapshots instead: {exc}"
            finally:
                if connection is not None:
                    connection.close()

        if frame is None:
            csv_path = self._find_csv_path(csv_candidates)
            if csv_path is None:
                raise FileNotFoundError(
                    f"Could not locate any table snapshot for {cache_key}: {', '.join(csv_candidates)}"
                )
            frame = pd.read_csv(csv_path)
            source = f"csv:{csv_path.name}"

        normalized = _normalise_columns(frame)
        self._frame_cache[cache_key] = normalized
        self._source_cache[cache_key] = source
        return normalized.copy()

    def load_sales_frame(self) -> pd.DataFrame:
        fact = self._load_table(
            "sales_fact",
            ["FactVentee", "FactVente", "factventee", "factvente"],
            ["FactVentee.csv", "Fact_Vente.csv"],
        )
        dim_date = self._load_table(
            "dim_date",
            ["DimDate", "dim_date", "dimdate"],
            ["dim_date.csv"],
        )
        dim_channel = self._load_table(
            "dim_channel",
            ["DimChannel", "DimCanal", "dim_channel", "dim_canal", "dimchannel"],
            ["dim_channel.csv"],
        )

        working = fact.copy()
        if "flag" in working.columns:
            working = working[working["flag"].fillna("").astype(str).str.lower() == "y"].copy()

        for column in ["datefk", "channelfk", "quantity", "price", "remise", "total_ht", "tva", "timbre", "total_ttc"]:
            if column not in working.columns:
                working[column] = 0

        for column in ["quantity", "price", "remise", "total_ht", "tva", "timbre", "total_ttc", "channelfk", "datefk"]:
            working[column] = _coerce_numeric(working[column])

        dim_date_pk = _pick_column(dim_date.columns, ["date_pk", "datepk", "id"])
        dim_date_value = _pick_column(dim_date.columns, ["date", "full_date", "fulldate"])
        if dim_date_pk and dim_date_value:
            date_map = dim_date[[dim_date_pk, dim_date_value]].rename(
                columns={dim_date_pk: "datefk", dim_date_value: "sale_date_dim"}
            )
            date_map["datefk"] = _coerce_numeric(date_map["datefk"])
            date_map["sale_date_dim"] = pd.to_datetime(date_map["sale_date_dim"], errors="coerce")
            working = working.merge(date_map, on="datefk", how="left")
        else:
            working["sale_date_dim"] = pd.NaT

        working["sale_date"] = working["sale_date_dim"].fillna(_parse_date_key(working["datefk"]))

        dim_channel_pk = _pick_column(dim_channel.columns, ["channelpk", "canalpk", "channel_fk", "id"])
        dim_channel_value = _pick_column(dim_channel.columns, ["channel", "canal", "channel_name", "name"])
        if dim_channel_pk and dim_channel_value and "channelfk" in working.columns:
            channel_map = dim_channel[[dim_channel_pk, dim_channel_value]].rename(
                columns={dim_channel_pk: "channelfk", dim_channel_value: "channel_name"}
            )
            channel_map["channelfk"] = _coerce_numeric(channel_map["channelfk"])
            working = working.merge(channel_map, on="channelfk", how="left")
        else:
            working["channel_name"] = np.nan

        working["channel_name"] = _clean_label(
            working["channel_name"].where(working["channel_name"].notna(), working["channelfk"].map(lambda value: f"Channel {int(value)}" if pd.notna(value) else "All channels")),
            fallback="All channels",
        )

        fallback_revenue = (working["price"].fillna(0) * working["quantity"].fillna(0) - working["remise"].fillna(0)).clip(lower=0)
        revenue = working["total_ttc"].fillna(0)
        working["revenue"] = revenue.where(revenue > 0, fallback_revenue)
        working["discount_value"] = working["remise"].fillna(0).clip(lower=0)

        working = working[working["sale_date"].notna()].copy()
        working = working[working["revenue"].notna()].copy()
        working["month"] = working["sale_date"].dt.month
        working["month_name"] = working["sale_date"].dt.month_name()
        working["weekday"] = working["sale_date"].dt.dayofweek
        working["weekday_name"] = working["sale_date"].dt.day_name()
        working["week_of_year"] = working["sale_date"].dt.isocalendar().week.astype(int)
        working["quarter"] = working["sale_date"].dt.quarter
        working["day"] = working["sale_date"].dt.day
        working["year"] = working["sale_date"].dt.year

        return working[
            [
                "sale_date",
                "year",
                "month",
                "month_name",
                "weekday",
                "weekday_name",
                "week_of_year",
                "quarter",
                "day",
                "channelfk",
                "channel_name",
                "quantity",
                "price",
                "discount_value",
                "revenue",
            ]
        ].rename(columns={"channelfk": "channel_fk"})

    def load_campaign_frame(self) -> pd.DataFrame:
        primary = self._load_table(
            "campaign_primary",
            ["FactCampaign", "factcampaign"],
            ["FactCampaign.csv"],
        )
        secondary = self._load_table(
            "campaign_secondary",
            ["fact_campaign"],
            ["fact_campaign.csv"],
        )
        dim_date = self._load_table(
            "dim_date",
            ["DimDate", "dim_date", "dimdate"],
            ["dim_date.csv"],
        )

        frames = []
        for frame in [primary, secondary]:
            working = frame.copy()
            rename_map = {}
            if "factcampaignpk" in working.columns:
                rename_map["factcampaignpk"] = "campaignpk"
            if "datefk" in working.columns:
                rename_map["datefk"] = "date_key"
            if "date_pk" in working.columns:
                rename_map["date_pk"] = "date_key"
            if "documentfk" in working.columns:
                rename_map["documentfk"] = "documentpk"
            working = working.rename(columns=rename_map)
            if "campaignpk" not in working.columns:
                working["campaignpk"] = np.arange(len(working))
            if "documentpk" not in working.columns:
                working["documentpk"] = np.arange(len(working))
            if "date_key" not in working.columns:
                working["date_key"] = np.nan
            for column in ["reach", "impressions", "frequency", "result", "views"]:
                if column not in working.columns:
                    working[column] = 0
                working[column] = _coerce_numeric(working[column]).fillna(0)
            if "price" not in working.columns:
                working["price"] = 0
            working["price"] = _currency_to_float(working["price"]).fillna(0)
            frames.append(working[["campaignpk", "date_key", "documentpk", "reach", "impressions", "frequency", "result", "price", "views"]])

        campaigns = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

        dim_date_pk = _pick_column(dim_date.columns, ["date_pk", "datepk", "id"])
        dim_date_value = _pick_column(dim_date.columns, ["date", "full_date", "fulldate"])
        if dim_date_pk and dim_date_value:
            date_map = dim_date[[dim_date_pk, dim_date_value]].rename(
                columns={dim_date_pk: "date_key", dim_date_value: "campaign_date_dim"}
            )
            date_map["date_key"] = _coerce_numeric(date_map["date_key"])
            date_map["campaign_date_dim"] = pd.to_datetime(date_map["campaign_date_dim"], errors="coerce")
            campaigns = campaigns.merge(date_map, on="date_key", how="left")
        else:
            campaigns["campaign_date_dim"] = pd.NaT

        campaigns["campaign_date"] = campaigns["campaign_date_dim"].fillna(_parse_date_key(campaigns["date_key"]))
        campaigns = campaigns[campaigns["campaign_date"].notna()].copy()
        campaigns["month"] = campaigns["campaign_date"].dt.month
        campaigns["weekday"] = campaigns["campaign_date"].dt.dayofweek
        campaigns["price"] = campaigns["price"].clip(lower=0)

        return campaigns[
            [
                "campaignpk",
                "documentpk",
                "campaign_date",
                "month",
                "weekday",
                "reach",
                "impressions",
                "frequency",
                "result",
                "price",
                "views",
            ]
        ]

    def load_supplier_frame(self) -> pd.DataFrame:
        purchases = self._load_table(
            "supplier_purchases",
            ["FactAchat", "factachat"],
            ["FactAchat.csv", "fact_achat.csv"],
        )
        suppliers = self._load_table(
            "dim_supplier",
            ["DimSupplier", "dim_supplier", "dimsupplier"],
            ["DimSupplier.csv", "dim_supplier.csv"],
        )
        addresses = self._load_table(
            "dim_address",
            ["DimAdress", "DimAdresse", "dim_adress", "dim_adresse", "dimadress"],
            ["dim_adress.csv", "DimAdress.csv", "DimAdresse.csv"],
        )

        working = purchases.copy()
        if "flag" in working.columns:
            working = working[working["flag"].fillna("").astype(str).str.lower() == "y"].copy()

        for column in ["supplierfk", "productfk", "quantity", "price", "remise", "total_ttc"]:
            if column not in working.columns:
                working[column] = 0
            working[column] = _coerce_numeric(working[column])

        suppliers = suppliers.rename(columns={"supplierpk": "supplierfk", "adressfk": "adressfk_dim"})
        addresses = addresses.rename(columns={"adresspk": "adressfk_dim"})

        working = working.merge(
            suppliers[["supplierfk", "supplier", "adressfk_dim"]],
            on="supplierfk",
            how="left",
        )
        working = working.merge(
            addresses[["adressfk_dim", "city", "governorate"]],
            on="adressfk_dim",
            how="left",
        )

        for column in ["supplier", "city", "governorate"]:
            if column not in working.columns:
                working[column] = "Unknown"
            working[column] = _clean_label(working[column])

        aggregated = (
            working.groupby(["supplierfk", "supplier", "governorate", "city"], dropna=False)
            .agg(
                total_quantity=("quantity", "sum"),
                total_spend=("total_ttc", "sum"),
                purchase_count=("supplierfk", "size"),
                average_unit_price=("price", "mean"),
                average_discount=("remise", "mean"),
                active_products=("productfk", "nunique"),
            )
            .reset_index()
        )

        aggregated["average_unit_price"] = aggregated["average_unit_price"].fillna(0).clip(lower=0)
        aggregated["spend_per_purchase"] = aggregated["total_spend"] / aggregated["purchase_count"].replace(0, np.nan)
        aggregated["spend_per_purchase"] = aggregated["spend_per_purchase"].fillna(0)

        return aggregated

    def load_competitor_frame(self) -> pd.DataFrame:
        fact = self._load_table(
            "competitor_fact",
            ["FactConcurrents", "factconcurrents"],
            ["FactConcurrents.csv", "fact_concurrent.csv"],
        )
        products = self._load_table(
            "dim_product",
            ["DimProduct", "dim_products", "dimproduct"],
            ["DimProduct.csv", "dim_products.csv"],
        )
        categories = self._load_table(
            "dim_category",
            ["dim_category", "DimCategory"],
            ["dim_category.csv"],
        )
        competitors = self._load_table(
            "dim_competitors",
            ["dim_concurrents", "DimConcurrents"],
            ["dim_concurrents.csv"],
        )

        working = fact.copy()
        if "flag" in working.columns:
            working = working[working["flag"].fillna("").astype(str).str.lower() == "y"].copy()

        working["productfk"] = _coerce_numeric(working.get("productfk", 0))
        working["concurrentfk"] = _coerce_numeric(working.get("concurrentfk", 0))
        working["price_product"] = _currency_to_float(working.get("price_product", pd.Series(dtype=float)))
        working.loc[working["price_product"] < 0, "price_product"] = np.nan

        products = products.rename(columns={"productpk": "productfk", "categoryfk": "categoryfk"})
        categories = categories.rename(columns={"categorypk": "categoryfk"})
        competitors = competitors.rename(columns={"concurrentpk": "concurrentfk"})

        merged = working.merge(products[["productfk", "product", "categoryfk"]], on="productfk", how="left")
        merged = merged.merge(categories[["categoryfk", "category"]], on="categoryfk", how="left")
        merged = merged.merge(competitors[["concurrentfk", "concurrent"]], on="concurrentfk", how="left")

        merged["product"] = _clean_label(merged.get("product", pd.Series(dtype=str)), fallback="")
        merged["category"] = _clean_label(merged.get("category", pd.Series(dtype=str)))
        merged["concurrent"] = _clean_label(merged.get("concurrent", pd.Series(dtype=str)))
        merged = merged[merged["product"] != ""].copy()

        aggregated = (
            merged.groupby(["product", "category"], dropna=False)
            .agg(
                avg_price=("price_product", "mean"),
                competitor_count=("concurrent", "nunique"),
                observation_count=("product", "size"),
                dominant_competitor=("concurrent", lambda series: series.mode().iloc[0] if not series.mode().empty else "Unknown"),
            )
            .reset_index()
        )
        aggregated["avg_price"] = aggregated["avg_price"].fillna(aggregated["avg_price"].median()).fillna(0)
        aggregated["competitor_count"] = aggregated["competitor_count"].fillna(0).astype(int)
        aggregated["observation_count"] = aggregated["observation_count"].fillna(0).astype(int)

        return aggregated

