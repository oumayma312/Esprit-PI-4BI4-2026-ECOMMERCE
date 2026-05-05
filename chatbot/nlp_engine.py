from __future__ import annotations

import re
from dataclasses import dataclass, field

from .utils import normalize_search_text, tokenize_query


@dataclass
class ParsedQuery:
    intent: str
    entities: dict = field(default_factory=dict)
    is_data_question: bool = False


class NLPEngine:
    def __init__(self, table_names: list[str] | None = None) -> None:
        self.table_names = table_names or []
        self.intent_keywords: dict[str, tuple[str, ...]] = {
            "greeting": ("hello", "hi", "hey", "bonjour", "salut"),
            "exit": ("bye", "goodbye", "quit", "exit", "au revoir"),
            "kpi_summary": (
                "kpi",
                "summary",
                "overview",
                "performance",
                "revenue",
                "sales summary",
                "synthese",
                "resume",
            ),
            "top_products": (
                "top products",
                "best sellers",
                "best products",
                "top product",
                "top produits",
                "meilleurs produits",
            ),
            "dataset_overview": (
                "warehouse",
                "datawarehouse",
                "dataset",
                "csv",
                "table",
                "tables",
                "data source",
                "schema",
                "column",
                "columns",
                "fichier",
                "donnees",
            ),
            "cheap_low_sales": (
                "low sales",
                "underperforming",
                "small budget",
                "budget friendly",
                "cheap",
                "promote",
                "petit budget",
                "peu vendus",
            ),
            "search_product": (
                "find product",
                "search product",
                "reference",
                "category",
                "find",
                "chercher",
                "produit",
            ),
            "strategy": (
                "strategy",
                "recommendation",
                "campaign",
                "marketing plan",
                "what should we do",
                "strategie",
                "recommandation",
                "campagne",
            ),
        }

    def parse(self, text: str) -> ParsedQuery:
        normalized = normalize_search_text(text)
        intent = self._detect_intent(normalized)
        entities = self._extract_entities(text)

        data_keywords = {
            "warehouse",
            "datawarehouse",
            "dataset",
            "csv",
            "table",
            "tables",
            "schema",
            "column",
            "columns",
            "sales",
            "product",
            "products",
            "campaign",
            "finance",
            "customer",
            "supplier",
            "analytics",
            "dashboard",
            "data",
            "donnees",
        }
        query_tokens = set(tokenize_query(text))
        normalized_no_spaces = normalized.replace(" ", "")
        table_hits = any(
            normalize_search_text(table_name).replace(" ", "") in normalized_no_spaces
            for table_name in self.table_names
        )
        is_data_question = bool(query_tokens & data_keywords) or table_hits or intent in {
            "kpi_summary",
            "top_products",
            "dataset_overview",
            "cheap_low_sales",
            "search_product",
            "strategy",
        }
        return ParsedQuery(intent=intent, entities=entities, is_data_question=is_data_question)

    def _detect_intent(self, normalized_text: str) -> str:
        if re.search(r"\b(top|best|leading)\s+\d*\s*(products?|sellers?)\b", normalized_text):
            return "top_products"
        if re.search(r"\b(kpi|metrics?|performance|revenue|sales summary)\b", normalized_text):
            return "kpi_summary"
        if re.search(r"\b(what|which|show|list)\s+tables?\b", normalized_text):
            return "dataset_overview"
        if re.search(r"\b(warehouse|datawarehouse|dataset|csv|schema|columns?)\b", normalized_text):
            return "dataset_overview"
        if re.search(r"\b(find|search|lookup)\b.*\b(product|reference|category)\b", normalized_text):
            return "search_product"
        if re.search(r"\b(small budget|budget friendly|underperforming|low sales|promote)\b", normalized_text):
            return "cheap_low_sales"
        if re.search(r"\b(strategy|recommendation|campaign|marketing plan)\b", normalized_text):
            return "strategy"

        scores: dict[str, int] = {}
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                normalized_keyword = normalize_search_text(keyword)
                if normalized_keyword and normalized_keyword in normalized_text:
                    scores[intent] = scores.get(intent, 0) + len(normalized_keyword.split())

        if scores:
            return max(scores, key=scores.get)

        if len(normalized_text.split()) > 3:
            return "general"

        return "greeting"

    @staticmethod
    def _extract_entities(text: str) -> dict:
        out: dict = {}
        normalized = normalize_search_text(text)

        if any(term in normalized for term in ("petit budget", "budget faible", "low budget", "small budget")):
            out["budget_small"] = True

        top_match = re.search(r"(?:top|best|meilleurs?)\s*(\d+)", normalized)
        if top_match:
            out["top_n"] = int(top_match.group(1))

        price_match = re.search(r"(\d+[\.,]?\d*)\s*(dt|tnd|\$|usd|eur|euro|euros)", normalized)
        if price_match:
            out["price_cap"] = float(price_match.group(1).replace(",", "."))

        reference_match = re.search(r"\breference\s+([\w\-_.]+)", normalized)
        if reference_match:
            out["query"] = reference_match.group(1).strip()

        product_match = re.search(
            r"(?:product|category|categorie|cat[ée]gorie)\s*(?:named|called|:|-)\s*([\w\s\-.]+)",
            normalized,
        )
        if product_match:
            value = product_match.group(1).strip()
            if value and len(value) > 1:
                out["query"] = value

        table_match = re.search(
            r"(?:csv|table|file|fichier|dataset)\s*(?:named|called|:|-)\s*([\w\s\-_.]+)",
            normalized,
        )
        if table_match:
            value = table_match.group(1).strip()
            if value and len(value) > 1:
                out["dataset_query"] = value

        quoted_match = re.search(r"['\"]([^'\"]{2,})['\"]", text)
        if quoted_match and "query" not in out:
            out["query"] = quoted_match.group(1).strip()

        return out
