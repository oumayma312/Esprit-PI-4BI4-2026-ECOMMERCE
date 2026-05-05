from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .utils import compact_whitespace, normalize_column_name, parse_mixed_number


@dataclass
class TableProfile:
    rows: int
    columns: int
    missing_ratio: float
    numeric_columns: int
    text_columns: int


@dataclass
class WarehouseData:
    products: pd.DataFrame
    sales: pd.DataFrame
    categories: pd.DataFrame
    campaigns: pd.DataFrame
    product_metrics: pd.DataFrame
    all_tables: Dict[str, pd.DataFrame]
    table_profiles: Dict[str, TableProfile]


class DataWarehouse:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    def load(self) -> WarehouseData:
        csv_paths = sorted(self.data_dir.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files were found in {self.data_dir}")

        all_tables = {path.name: self._read_csv(path) for path in csv_paths}

        product_file = self._pick_table_name(["dim_products.csv", "dimproduct.csv"], all_tables)
        sales_file = self._pick_table_name(["fact_vente.csv", "factventee.csv"], all_tables)
        category_file = self._pick_table_name(["dim_category.csv"], all_tables, required=False)
        campaign_file = self._pick_table_name(["fact_campaign.csv"], all_tables, required=False)

        products = all_tables.get(product_file, pd.DataFrame())
        sales = all_tables.get(sales_file, pd.DataFrame())
        categories = all_tables.get(category_file, pd.DataFrame()) if category_file else pd.DataFrame()
        campaigns = all_tables.get(campaign_file, pd.DataFrame()) if campaign_file else pd.DataFrame()

        products = self._normalize_products(products)
        sales = self._normalize_sales(sales)
        categories = self._normalize_categories(categories)
        campaigns = self._normalize_campaigns(campaigns)

        product_metrics = self._build_product_metrics(products, sales, categories)
        table_profiles = {name: self._profile_table(df) for name, df in all_tables.items()}

        return WarehouseData(
            products=products,
            sales=sales,
            categories=categories,
            campaigns=campaigns,
            product_metrics=product_metrics,
            all_tables=all_tables,
            table_profiles=table_profiles,
        )

    def _pick_table_name(
        self, preferred_names: list[str], all_tables: Dict[str, pd.DataFrame], required: bool = True
    ) -> Optional[str]:
        existing_files = {name.lower(): name for name in all_tables}

        for name in preferred_names:
            if name.lower() in existing_files:
                return existing_files[name.lower()]

        for file_name, real_name in existing_files.items():
            for expected in preferred_names:
                expected_base = expected.replace("_", "").replace("-", "").replace(".csv", "")
                candidate_base = file_name.replace("_", "").replace("-", "").replace(".csv", "")
                if expected_base in candidate_base:
                    return real_name

        if required:
            raise FileNotFoundError(
                f"No matching file was found for {preferred_names} in {self.data_dir}"
            )
        return None

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        for encoding in ("utf-8", "latin-1"):
            try:
                frame = pd.read_csv(path, encoding=encoding)
                frame.columns = [normalize_column_name(c) for c in frame.columns]
                return frame
            except UnicodeDecodeError:
                continue
        frame = pd.read_csv(path)
        frame.columns = [normalize_column_name(c) for c in frame.columns]
        return frame

    @staticmethod
    def _normalize_products(products: pd.DataFrame) -> pd.DataFrame:
        if products.empty:
            return pd.DataFrame(columns=["productpk", "reference", "product", "categoryfk"])

        out = products.copy()

        if "reference" not in out.columns and "productcode" in out.columns:
            out["reference"] = out["productcode"]
        if "product" not in out.columns:
            out["product"] = out.get("reference", "Unknown product")
        if "categoryfk" not in out.columns:
            out["categoryfk"] = None
        if "productpk" not in out.columns:
            return pd.DataFrame(columns=["productpk", "reference", "product", "categoryfk"])

        if "reference" in out.columns:
            out["reference"] = out["reference"].fillna("").map(compact_whitespace)
        if "product" in out.columns:
            out["product"] = out["product"].fillna("").map(compact_whitespace)

        out["productpk"] = pd.to_numeric(out.get("productpk"), errors="coerce")
        out = out.dropna(subset=["productpk"]).copy()
        out["productpk"] = out["productpk"].astype(int)
        return out

    @staticmethod
    def _normalize_sales(sales: pd.DataFrame) -> pd.DataFrame:
        if sales.empty:
            return pd.DataFrame(columns=["productfk", "quantity", "total_ttc"])

        out = sales.copy()

        if "productfk" not in out.columns:
            out["productfk"] = out.get("product_pk")
        if "quantity" not in out.columns:
            out["quantity"] = 1
        if "total_ttc" not in out.columns:
            out["total_ttc"] = out.get("amount")

        out["productfk"] = pd.to_numeric(out.get("productfk"), errors="coerce")
        out["quantity"] = out["quantity"].apply(parse_mixed_number)
        out["total_ttc"] = out["total_ttc"].apply(parse_mixed_number)

        out = out.dropna(subset=["productfk"]).copy()
        out["productfk"] = out["productfk"].astype(int)
        out["quantity"] = out["quantity"].fillna(1.0)
        out["total_ttc"] = out["total_ttc"].fillna(0.0)

        return out

    @staticmethod
    def _normalize_categories(categories: pd.DataFrame) -> pd.DataFrame:
        if categories.empty:
            return categories

        out = categories.copy()
        if "categorypk" not in out.columns:
            return pd.DataFrame(columns=["categorypk", "category"])

        out["categorypk"] = pd.to_numeric(out["categorypk"], errors="coerce")
        out = out.dropna(subset=["categorypk"]).copy()
        out["categorypk"] = out["categorypk"].astype(int)
        if "category" not in out.columns:
            out["category"] = "Uncategorized"
        out["category"] = out["category"].fillna("Uncategorized").map(compact_whitespace)
        return out[["categorypk", "category"]]

    @staticmethod
    def _normalize_campaigns(campaigns: pd.DataFrame) -> pd.DataFrame:
        if campaigns.empty:
            return campaigns

        out = campaigns.copy()
        numeric_targets = ["reach", "impressions", "frequency", "result", "price", "views"]
        for col in numeric_targets:
            if col in out.columns:
                out[col] = out[col].apply(parse_mixed_number)
        return out

    @staticmethod
    def _build_product_metrics(
        products: pd.DataFrame, sales: pd.DataFrame, categories: pd.DataFrame
    ) -> pd.DataFrame:
        empty_columns = [
            "productpk",
            "reference",
            "product",
            "categoryfk",
            "quantity_sold",
            "revenue",
            "orders",
            "avg_ticket",
            "unit_price_est",
            "category",
        ]

        if products.empty:
            return pd.DataFrame(columns=empty_columns)

        grouped = (
            sales.groupby("productfk", as_index=False)
            .agg(
                quantity_sold=("quantity", "sum"),
                revenue=("total_ttc", "sum"),
                orders=("productfk", "count"),
            )
            .rename(columns={"productfk": "productpk"})
        )

        metrics = products.merge(grouped, on="productpk", how="left")
        for col in ["quantity_sold", "revenue", "orders"]:
            metrics[col] = pd.to_numeric(metrics[col], errors="coerce").fillna(0.0)

        metrics["orders"] = metrics["orders"].astype(int)

        avg_ticket_raw = metrics["revenue"] / metrics["orders"].replace(0, float("nan"))
        unit_price_raw = metrics["revenue"] / metrics["quantity_sold"].replace(0, float("nan"))
        metrics["avg_ticket"] = pd.to_numeric(avg_ticket_raw, errors="coerce").fillna(0.0)
        metrics["unit_price_est"] = pd.to_numeric(unit_price_raw, errors="coerce").fillna(0.0)

        if not categories.empty and "categoryfk" in metrics.columns:
            metrics = metrics.merge(
                categories,
                left_on="categoryfk",
                right_on="categorypk",
                how="left",
            )
            metrics["category"] = metrics["category"].fillna("Uncategorized")
        else:
            metrics["category"] = "Uncategorized"

        return metrics.sort_values(by="revenue", ascending=False).reset_index(drop=True)

    @staticmethod
    def _profile_table(table: pd.DataFrame) -> TableProfile:
        rows = int(len(table))
        columns = int(len(table.columns))

        if rows == 0 or columns == 0:
            return TableProfile(
                rows=rows,
                columns=columns,
                missing_ratio=0.0,
                numeric_columns=0,
                text_columns=0,
            )

        missing_ratio = float(table.isna().sum().sum() / (rows * columns))
        numeric_columns = int(table.select_dtypes(include="number").shape[1])
        text_columns = int(table.select_dtypes(exclude="number").shape[1])

        return TableProfile(
            rows=rows,
            columns=columns,
            missing_ratio=missing_ratio,
            numeric_columns=numeric_columns,
            text_columns=text_columns,
        )
