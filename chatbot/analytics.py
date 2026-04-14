from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_loader import WarehouseData


@dataclass
class InsightBundle:
    summary: str
    table: pd.DataFrame


class MarketingAnalytics:
    def __init__(self, data: WarehouseData) -> None:
        self.data = data
        self.metrics = data.product_metrics.copy()

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
                f"{name}: {profile.rows} lignes, {profile.columns} colonnes, "
                f"manquants {profile.missing_ratio * 100:.1f}%"
            )

        detail_str = " | ".join(details) if details else "Aucun detail disponible"
        return (
            f"CSV detectes: {total_tables} | Lignes totales: {total_rows} | Colonnes totales: {total_cols}. "
            f"Principales tables: {detail_str}."
        )

    def search_dataset(self, query: str, n: int = 8) -> InsightBundle:
        q = query.strip().lower()
        rows = []

        for file_name, table in self.data.all_tables.items():
            profile = self.data.table_profiles[file_name]
            columns_text = ", ".join(map(str, table.columns.tolist()))
            haystack = f"{file_name} {columns_text}".lower()
            if q and q not in haystack:
                continue

            rows.append(
                {
                    "table": file_name,
                    "rows": profile.rows,
                    "columns": profile.columns,
                    "missing_pct": round(profile.missing_ratio * 100, 2),
                    "sample_columns": columns_text[:140],
                }
            )

        if not rows:
            return InsightBundle(
                summary=f"Aucun fichier CSV ne correspond a '{query}'.",
                table=pd.DataFrame(columns=["table", "rows", "columns", "missing_pct", "sample_columns"]),
            )

        result = pd.DataFrame(rows).sort_values(by=["rows", "columns"], ascending=[False, False]).head(n)
        return InsightBundle(
            summary=f"{len(result)} fichier(s) CSV correspondent a '{query}'.",
            table=result,
        )

    def kpi_summary(self) -> str:
        if self.metrics.empty:
            return "Aucun jeu produits/ventes exploitable detecte pour calculer les KPI marketing."

        total_products = len(self.metrics)
        sold_products = int((self.metrics["orders"] > 0).sum())
        total_revenue = float(self.metrics["revenue"].sum())
        total_qty = float(self.metrics["quantity_sold"].sum())

        top = self.metrics.head(3)[["product", "revenue"]]
        top_str = ", ".join(
            f"{row.product} ({row.revenue:.2f})" for row in top.itertuples(index=False)
        )

        return (
            f"Produits: {total_products} | Produits vendus: {sold_products} | "
            f"CA estime: {total_revenue:.2f} | Quantite vendue: {total_qty:.0f}. "
            f"Top produits: {top_str if top_str else 'n/a'}."
        )

    def top_products(self, n: int = 5) -> InsightBundle:
        if self.metrics.empty:
            return InsightBundle(
                summary="Impossible de produire un top produits: donnees produits/ventes indisponibles.",
                table=pd.DataFrame(columns=["product", "category", "quantity_sold", "orders", "revenue", "unit_price_est"]),
            )

        n = max(1, min(n, 20))
        table = self.metrics.nlargest(n, "revenue")[
            ["product", "category", "quantity_sold", "orders", "revenue", "unit_price_est"]
        ]
        summary = f"Top {n} produits par revenu identifies."
        return InsightBundle(summary=summary, table=table)

    def cheap_low_sales(self, n: int = 8, price_cap: float | None = None) -> InsightBundle:
        if self.metrics.empty:
            return InsightBundle(
                summary="Impossible de detecter les opportunites: donnees ventes indisponibles.",
                table=pd.DataFrame(columns=["product", "category", "quantity_sold", "revenue", "unit_price_est"]),
            )

        frame = self.metrics.copy()
        frame = frame[frame["orders"] > 0]

        if frame.empty:
            return InsightBundle(
                summary="Aucune vente exploitable pour detecter les opportunites.",
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

        table = candidates.sort_values(
            by=["quantity_sold", "unit_price_est"], ascending=[True, True]
        ).head(n)[["product", "category", "quantity_sold", "revenue", "unit_price_est"]]

        summary = (
            "Produits a faible volume et prix accessible identifies pour une campagne budget limite."
        )
        return InsightBundle(summary=summary, table=table)

    def find_product(self, query: str, n: int = 5) -> InsightBundle:
        if self.metrics.empty:
            return InsightBundle(
                summary="Recherche produit indisponible: aucune table produits/ventes valide detectee.",
                table=pd.DataFrame(columns=["product", "category", "quantity_sold", "orders", "revenue", "unit_price_est"]),
            )

        q = query.strip().lower()
        if not q:
            return InsightBundle(summary="Precise un nom ou mot-cle produit.", table=self.metrics.head(0))

        frame = self.metrics.copy()
        mask = frame["product"].astype(str).str.lower().str.contains(q, na=False)
        if "reference" in frame.columns:
            mask = mask | frame["reference"].astype(str).str.lower().str.contains(q, na=False)
        if "category" in frame.columns:
            mask = mask | frame["category"].astype(str).str.lower().str.contains(q, na=False)

        result = frame[mask].head(n)[
            ["product", "category", "quantity_sold", "orders", "revenue", "unit_price_est"]
        ]

        if result.empty:
            return InsightBundle(summary=f"Aucun produit trouve pour '{query}'.", table=result)

        return InsightBundle(summary=f"{len(result)} produit(s) correspondant a '{query}'.", table=result)

    def strategy_recommendations(self, budget_small: bool = False) -> str:
        top = self.metrics.nlargest(3, "revenue")
        weak = self.cheap_low_sales(n=3).table

        top_names = ", ".join(self._clean_label(x) for x in top["product"].tolist()) if not top.empty else "n/a"
        weak_names = ", ".join(self._clean_label(x) for x in weak["product"].tolist()) if not weak.empty else "n/a"

        budget_line = (
            "Prioriser des contenus UGC et des campagnes retargeting a petit budget." if budget_small
            else "Combiner acquisition payante et retargeting sur les segments chauds."
        )

        return (
            f"1) Capitaliser sur les best-sellers ({top_names}) avec bundles et upsell. "
            f"2) Relancer les produits sous-exploites ({weak_names}) via offres flash et preuve sociale. "
            f"3) {budget_line} "
            "4) Tester 2-3 creatives par produit et couper les campagnes sous le seuil de rentabilite."
        )

    @staticmethod
    def _clean_label(value: str, max_len: int = 70) -> str:
        text = " ".join(str(value).split())
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."


def format_table_for_chat(table: pd.DataFrame, max_rows: int = 8) -> str:
    if table.empty:
        return "(Aucune ligne)"

    preview = table.head(max_rows).copy()
    for col in ["revenue", "unit_price_est", "quantity_sold", "orders"]:
        if col in preview.columns:
            preview[col] = preview[col].astype(float).round(2)

    return preview.to_string(index=False)
