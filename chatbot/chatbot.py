from __future__ import annotations

from .analytics import MarketingAnalytics, format_table_for_chat
from .config import AppConfig
from .data_loader import DataWarehouse
from .gemini_client import GeminiClient
from .nlp_engine import NLPEngine


class MarketingChatbot:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        warehouse = DataWarehouse(config.data_dir)
        data = warehouse.load()

        self.analytics = MarketingAnalytics(data)
        self.nlp = NLPEngine()
        self.gemini = GeminiClient(config.gemini_api_key, config.gemini_model)

    def respond(self, message: str) -> str:
        parsed = self.nlp.parse(message)
        entities = parsed.entities

        if parsed.intent == "exit":
            return "Merci. A bientot."

        csv_related_intents = {
            "kpi_summary",
            "dataset_overview",
            "top_products",
            "cheap_low_sales",
            "search_product",
            "strategy",
        }
        csv_related = parsed.intent in csv_related_intents

        if parsed.intent == "greeting":
            base_answer = (
                "Bonjour. Je peux analyser tes CSV et te proposer des recommandations marketing actionnables. "
                "Tu peux demander: top produits, produits a promouvoir avec petit budget, ou strategie globale."
            )
        elif parsed.intent == "kpi_summary":
            base_answer = self.analytics.kpi_summary()
        elif parsed.intent == "dataset_overview":
            dataset_query = entities.get("dataset_query")
            if dataset_query:
                bundle = self.analytics.search_dataset(query=dataset_query, n=10)
                base_answer = f"{bundle.summary}\n{format_table_for_chat(bundle.table, max_rows=10)}"
            else:
                base_answer = self.analytics.dataset_overview()
        elif parsed.intent == "top_products":
            n = entities.get("top_n", 5)
            bundle = self.analytics.top_products(n=n)
            base_answer = f"{bundle.summary}\n{format_table_for_chat(bundle.table)}"
        elif parsed.intent == "cheap_low_sales":
            bundle = self.analytics.cheap_low_sales(
                n=entities.get("top_n", 8),
                price_cap=entities.get("price_cap"),
            )
            base_answer = f"{bundle.summary}\n{format_table_for_chat(bundle.table)}"
        elif parsed.intent == "search_product":
            query = entities.get("query")
            if not query:
                query = message
            bundle = self.analytics.find_product(query=query, n=8)
            base_answer = f"{bundle.summary}\n{format_table_for_chat(bundle.table)}"
        elif parsed.intent == "strategy":
            base_answer = self.analytics.strategy_recommendations(
                budget_small=bool(entities.get("budget_small"))
            )
        else:
            base_answer = self.analytics.strategy_recommendations(
                budget_small=bool(entities.get("budget_small"))
            )

        csv_context_for_llm = self._compact_context(base_answer)
        gemini_first = self.gemini.generate_initial_answer(user_message=message)

        if gemini_first:
            hybrid = self.gemini.compose_hybrid_answer(
                user_message=message,
                gemini_draft=gemini_first,
                csv_context=csv_context_for_llm,
                prioritize_csv=csv_related,
            )
            if hybrid:
                return hybrid

            if csv_related:
                return f"{gemini_first}\n\nDonnees CSV pertinentes:\n{csv_context_for_llm}"

            return gemini_first

        gemini_status = self.gemini.status_message()
        if not self.gemini.available and self.config.gemini_enabled:
            return f"{base_answer}\n\nNote Gemini: {gemini_status}"

        return base_answer

    @staticmethod
    def _compact_context(text: str, max_chars: int = 1800) -> str:
        normalized = " ".join(str(text).split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3] + "..."
