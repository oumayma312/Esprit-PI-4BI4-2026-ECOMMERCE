from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_loader import WarehouseData
from .utils import compact_whitespace, normalize_search_text, tokenize_query


@dataclass
class InsightBundle:
    summary: str
    table: pd.DataFrame


class MarketingAnalytics:
    def __init__(self, data: WarehouseData) -> None:
        self.data = data
        self.metrics = data.product_metrics.copy()
        self.table_catalog = self._build_table_catalog()

    def dataset_overview(self) -> str:
        total_tables = len(self.data.all_tables)
        total_rows = sum(profile.rows for profile in self.data.table_profiles.values())
        total_cols = sum(profile.columns for profile in self.data.table_profiles.values())

        ranked = sorted(
            self.data.table_profiles.items(),
            key=lambda item: item[1].rows,
            reverse=True,
        )[:6]

        details = []
        for name, profile in ranked:
            details.append(
                f"{name}: {profile.rows} rows, {profile.columns} columns, "
                f"missing values {profile.missing_ratio * 100:.1f}%"
            )

        detail_str = " | ".join(details) if details else "No details available"
        return (
            f"Detected CSV files: {total_tables} | Total rows: {total_rows} | Total columns: {total_cols}. "
            f"Largest tables: {detail_str}."
        )

    def search_dataset(self, query: str, n: int = 8) -> InsightBundle:
        matches = self._rank_table_matches(query)[: max(1, min(n, 12))]

        if not matches:
            return InsightBundle(
                summary=f"No CSV file matched '{query}'.",
                table=pd.DataFrame(columns=["table", "rows", "columns", "missing_pct", "sample_columns"]),
            )

        rows = [
            {
                "table": match["table"],
                "rows": match["rows"],
                "columns": match["columns"],
                "missing_pct": round(match["missing_ratio"] * 100, 2),
                "sample_columns": match["sample_columns"],
            }
            for match in matches
        ]
        result = pd.DataFrame(rows)
        return InsightBundle(
            summary=f"{len(result)} CSV file(s) matched '{query}'.",
            table=result,
        )

    def kpi_summary(self) -> str:
        if self.metrics.empty:
            return "No usable product and sales tables were detected, so marketing KPIs are unavailable."

        total_products = len(self.metrics)
        sold_products = int((self.metrics["orders"] > 0).sum())
        total_revenue = float(self.metrics["revenue"].sum())
        total_qty = float(self.metrics["quantity_sold"].sum())

        top = self.metrics.head(3)[["product", "revenue"]]
        top_str = ", ".join(
            f"{self._clean_label(row.product, max_len=60)} ({row.revenue:.2f})"
            for row in top.itertuples(index=False)
        )

        return (
            f"Products: {total_products} | Sold products: {sold_products} | "
            f"Estimated revenue: {total_revenue:.2f} | Units sold: {total_qty:.0f}. "
            f"Top products: {top_str if top_str else 'n/a'}."
        )

    def top_products(self, n: int = 5) -> InsightBundle:
        if self.metrics.empty:
            return InsightBundle(
                summary="Top products are unavailable because product or sales data could not be loaded.",
                table=pd.DataFrame(columns=["product", "category", "quantity_sold", "orders", "revenue", "unit_price_est"]),
            )

        n = max(1, min(n, 20))
        table = self._prepare_product_table(
            self.metrics.nlargest(n, "revenue")[
                ["product", "category", "quantity_sold", "orders", "revenue", "unit_price_est"]
            ]
        )
        summary = f"Top {n} products by revenue identified."
        return InsightBundle(summary=summary, table=table)

    def cheap_low_sales(self, n: int = 8, price_cap: float | None = None) -> InsightBundle:
        if self.metrics.empty:
            return InsightBundle(
                summary="Opportunity detection is unavailable because sales data could not be loaded.",
                table=pd.DataFrame(columns=["product", "category", "quantity_sold", "revenue", "unit_price_est"]),
            )

        frame = self.metrics.copy()
        frame = frame[frame["orders"] > 0]

        if frame.empty:
            return InsightBundle(
                summary="No usable sales rows were found for opportunity detection.",
                table=frame,
            )

        if price_cap is None:
            price_cap = float(frame["unit_price_est"].quantile(0.35))

        sales_threshold = float(frame["quantity_sold"].quantile(0.3))
        candidates = frame[
            (frame["unit_price_est"] <= price_cap) & (frame["quantity_sold"] <= sales_threshold)
        ]

        if candidates.empty:
            candidates = frame.nsmallest(n, "quantity_sold")

        table = self._prepare_product_table(
            candidates.sort_values(
                by=["quantity_sold", "unit_price_est"], ascending=[True, True]
            ).head(n)[["product", "category", "quantity_sold", "revenue", "unit_price_est"]]
        )

        summary = "Low-volume, budget-friendly products were identified for a low-cost promotion."
        return InsightBundle(summary=summary, table=table)

    def find_product(self, query: str, n: int = 5) -> InsightBundle:
        if self.metrics.empty:
            return InsightBundle(
                summary="Product search is unavailable because valid product and sales tables were not detected.",
                table=pd.DataFrame(columns=["product", "category", "quantity_sold", "orders", "revenue", "unit_price_est"]),
            )

        q = normalize_search_text(query)
        if not q:
            return InsightBundle(summary="Please provide a product name, reference, or keyword.", table=self.metrics.head(0))

        frame = self.metrics.copy()
        normalized_product = frame["product"].astype(str).map(normalize_search_text)
        mask = normalized_product.str.contains(q, na=False)
        if "reference" in frame.columns:
            mask = mask | frame["reference"].astype(str).map(normalize_search_text).str.contains(q, na=False)
        if "category" in frame.columns:
            mask = mask | frame["category"].astype(str).map(normalize_search_text).str.contains(q, na=False)

        result = self._prepare_product_table(
            frame[mask].head(n)[
                ["product", "category", "quantity_sold", "orders", "revenue", "unit_price_est"]
            ]
        )

        if result.empty and len(q.split()) > 1:
            for token in q.split():
                token_mask = normalized_product.str.contains(token, na=False)
                if "reference" in frame.columns:
                    token_mask = token_mask | frame["reference"].astype(str).map(normalize_search_text).str.contains(token, na=False)
                if "category" in frame.columns:
                    token_mask = token_mask | frame["category"].astype(str).map(normalize_search_text).str.contains(token, na=False)
                result = self._prepare_product_table(
                    frame[token_mask].head(n)[
                        ["product", "category", "quantity_sold", "orders", "revenue", "unit_price_est"]
                    ]
                )
                if not result.empty:
                    break

        if result.empty:
            return InsightBundle(summary=f"No product matched '{query}'.", table=result)

        return InsightBundle(summary=f"{len(result)} product(s) matched '{query}'.", table=result)

    def strategy_recommendations(self, budget_small: bool = False) -> str:
        top = self.metrics.nlargest(3, "revenue")
        weak = self.cheap_low_sales(n=3).table

        top_names = ", ".join(self._clean_label(x) for x in top["product"].tolist()) if not top.empty else "n/a"
        weak_names = ", ".join(self._clean_label(x) for x in weak["product"].tolist()) if not weak.empty else "n/a"

        budget_line = (
            "Prioritize UGC-style content and low-cost retargeting campaigns." if budget_small
            else "Blend paid acquisition with retargeting on the warmest segments."
        )

        return (
            f"1) Double down on the best sellers ({top_names}) with bundles and upsells. "
            f"2) Reactivate underused products ({weak_names}) with flash offers and social proof. "
            f"3) {budget_line} "
            "4) Test 2 to 3 creatives per product and pause campaigns below your profitability threshold."
        )

    def campaign_summary(self) -> str:
        campaigns = self.data.campaigns
        if campaigns.empty:
            return "Campaign metrics are not available in the warehouse."

        totals = {}
        for column in ("reach", "impressions", "result", "views"):
            if column in campaigns.columns:
                totals[column] = float(pd.to_numeric(campaigns[column], errors="coerce").fillna(0).sum())
        avg_frequency = 0.0
        if "frequency" in campaigns.columns:
            avg_frequency = float(pd.to_numeric(campaigns["frequency"], errors="coerce").fillna(0).mean())

        return (
            f"Campaign snapshot: reach {totals.get('reach', 0):.0f}, "
            f"impressions {totals.get('impressions', 0):.0f}, "
            f"results {totals.get('result', 0):.0f}, "
            f"views {totals.get('views', 0):.0f}, "
            f"average frequency {avg_frequency:.2f}."
        )

    def relevant_context(
        self,
        query: str,
        product_hint: str | None = None,
        dataset_hint: str | None = None,
        limit: int = 4,
    ) -> tuple[str, list[str]]:
        sections: list[str] = [self.kpi_summary()]
        sources: list[str] = []
        normalized_query = normalize_search_text(query)
        table_request_keywords = {
            "warehouse",
            "datawarehouse",
            "table",
            "tables",
            "csv",
            "dataset",
            "schema",
            "column",
            "columns",
            "customer",
            "supplier",
            "document",
            "campaign",
            "finance",
            "shipping",
            "sales",
        }
        generic_query_words = {
            "what",
            "which",
            "show",
            "list",
            "named",
            "called",
            "are",
            "the",
            "in",
            "for",
            "me",
            "all",
        }
        query_tokens = set(tokenize_query(query))
        specific_tokens = query_tokens - table_request_keywords - generic_query_words

        if any(token in normalized_query for token in ("campaign", "ads", "marketing", "promotion", "promote")):
            sections.append(self.campaign_summary())
            if not self.data.campaigns.empty:
                sources.append(self._actual_table_name("fact_campaign.csv"))

        if product_hint:
            product_bundle = self.find_product(product_hint, n=5)
            if not product_bundle.table.empty:
                sections.append(
                    f"Product lookup for '{product_hint}':\n{format_table_for_chat(product_bundle.table, max_rows=5)}"
                )
                sources.append("product_metrics")

        if dataset_hint or specific_tokens:
            dataset_bundle = self.search_dataset(dataset_hint or query, n=limit)
            for table_name in dataset_bundle.table["table"].tolist():
                descriptor = next((item for item in self.table_catalog if item["table"] == table_name), None)
                if descriptor is None:
                    continue
                sections.append(
                    f"Table {table_name}: {descriptor['rows']} rows, {descriptor['columns']} columns. "
                    f"Columns: {descriptor['columns_text']}. "
                    f"Sample rows:\n{descriptor['sample_rows']}"
                )
                sources.append(table_name)

        unique_sources = list(dict.fromkeys(source for source in sources if source))
        unique_sections = [compact_whitespace(section) if "\n" not in section else section for section in sections if section]
        return "\n\n".join(unique_sections[: limit + 2]), unique_sources

    def _build_table_catalog(self) -> list[dict]:
        catalog: list[dict] = []
        for file_name, table in self.data.all_tables.items():
            profile = self.data.table_profiles[file_name]
            columns = [str(column) for column in table.columns.tolist()]
            preview = table.head(3).copy()
            preview = preview.iloc[:, : min(len(preview.columns), 6)]
            preview = preview.fillna("").astype(str)
            preview_text = preview.to_string(index=False) if not preview.empty else "(No sample rows)"
            catalog.append(
                {
                    "table": file_name,
                    "rows": profile.rows,
                    "columns": profile.columns,
                    "missing_ratio": profile.missing_ratio,
                    "columns_text": ", ".join(columns[:10]),
                    "sample_columns": ", ".join(columns[:6]),
                    "sample_rows": preview_text,
                    "search_text": normalize_search_text(f"{file_name} {' '.join(columns)} {preview_text}"),
                }
            )
        return catalog

    def _rank_table_matches(self, query: str) -> list[dict]:
        tokens = tokenize_query(query)
        if not tokens:
            return sorted(
                self.table_catalog,
                key=lambda item: (item["rows"], item["columns"]),
                reverse=True,
            )

        ranked: list[dict] = []
        query_text = normalize_search_text(query)
        for descriptor in self.table_catalog:
            score = 0
            file_name = normalize_search_text(descriptor["table"])
            columns_text = normalize_search_text(descriptor["columns_text"])
            search_text = descriptor["search_text"]
            if query_text and query_text in search_text:
                score += 8
            for token in tokens:
                if token in file_name:
                    score += 5
                if token in columns_text:
                    score += 3
                if token in search_text:
                    score += 1
            if score > 0:
                ranked.append({**descriptor, "score": score})

        exact_name_matches = [
            item for item in ranked if query_text and query_text in normalize_search_text(item["table"])
        ]
        if exact_name_matches:
            exact_name_matches.sort(
                key=lambda item: (item["score"], item["rows"], item["columns"]),
                reverse=True,
            )
            return exact_name_matches

        ranked.sort(key=lambda item: (item["score"], item["rows"], item["columns"]), reverse=True)
        return ranked

    def _actual_table_name(self, preferred_name: str) -> str:
        preferred = normalize_search_text(preferred_name)
        for name in self.data.all_tables:
            if normalize_search_text(name) == preferred:
                return name
        return preferred_name

    def _prepare_product_table(self, table: pd.DataFrame) -> pd.DataFrame:
        if table.empty:
            return table

        prepared = table.copy()
        if "product" in prepared.columns:
            prepared["product"] = prepared["product"].astype(str).map(lambda value: self._clean_label(value, max_len=82))
        if "category" in prepared.columns:
            prepared["category"] = prepared["category"].astype(str).map(lambda value: self._clean_label(value, max_len=42))
        return prepared

    @staticmethod
    def _clean_label(value: str, max_len: int = 70) -> str:
        text = compact_whitespace(value)
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."


def format_table_for_chat(table: pd.DataFrame, max_rows: int = 8) -> str:
    if table.empty:
        return "(No rows)"

    preview = table.head(max_rows).copy()
    for col in ["revenue", "unit_price_est", "quantity_sold", "orders"]:
        if col in preview.columns:
            preview[col] = preview[col].astype(float).round(2)

    return preview.to_string(index=False)
