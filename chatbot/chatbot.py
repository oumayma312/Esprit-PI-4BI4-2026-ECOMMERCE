from __future__ import annotations

from dataclasses import dataclass

from .analytics import MarketingAnalytics, format_table_for_chat
from .config import AppConfig
from .data_loader import DataWarehouse
from .gemini_client import GeminiClient
from .nlp_engine import NLPEngine
from .utils import compact_whitespace


@dataclass
class ChatbotReply:
    answer: str
    mode: str
    sources: list[str]
    model: str


class MarketingChatbot:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        warehouse = DataWarehouse(config.data_dir)
        data = warehouse.load()

        self.analytics = MarketingAnalytics(data)
        self.nlp = NLPEngine(table_names=list(data.all_tables.keys()))
        self.gemini = GeminiClient(
            config.gemini_api_key,
            config.gemini_model,
            timeout_seconds=config.request_timeout_seconds,
        )
        self.system_instruction = (
            "You are Story AI, a concise and helpful analytics assistant for a Power BI dashboard. "
            "Always answer in English. "
            "You can handle general questions, but when warehouse context is provided you must treat it as the source of truth for facts, numbers, and table details. "
            "Never invent metrics, product names, or schema details that are not present in the provided context. "
            "If the data is incomplete, say so clearly and suggest the next best question. "
            "Prefer compact answers with direct recommendations when the user asks for strategy."
        )

    def respond(self, message: str, history: list[dict] | None = None) -> ChatbotReply:
        user_message = compact_whitespace(message)
        if not user_message:
            return ChatbotReply(
                answer="Ask me anything about the dashboards, products, campaigns, or warehouse tables.",
                mode="local",
                sources=[],
                model="local",
            )

        history = history or []
        parsed = self.nlp.parse(user_message)
        entities = parsed.entities

        if parsed.intent == "exit":
            return ChatbotReply(
                answer="Thanks for chatting. I'll be here whenever you need another data check.",
                mode="local",
                sources=[],
                model="local",
            )

        if parsed.intent == "greeting":
            local_answer = (
                "Hello. I can answer general questions, explore the warehouse CSV files, "
                "and turn product, sales, and campaign data into quick recommendations."
            )
        elif parsed.intent == "kpi_summary":
            local_answer = self.analytics.kpi_summary()
        elif parsed.intent == "dataset_overview":
            dataset_query = entities.get("dataset_query")
            if dataset_query:
                bundle = self.analytics.search_dataset(query=dataset_query, n=10)
                local_answer = f"{bundle.summary}\n{format_table_for_chat(bundle.table, max_rows=10)}"
            else:
                local_answer = self.analytics.dataset_overview()
        elif parsed.intent == "top_products":
            n = entities.get("top_n", 5)
            bundle = self.analytics.top_products(n=n)
            local_answer = f"{bundle.summary}\n{format_table_for_chat(bundle.table)}"
        elif parsed.intent == "cheap_low_sales":
            bundle = self.analytics.cheap_low_sales(
                n=entities.get("top_n", 8),
                price_cap=entities.get("price_cap"),
            )
            local_answer = f"{bundle.summary}\n{format_table_for_chat(bundle.table)}"
        elif parsed.intent == "search_product":
            query = entities.get("query") or user_message
            bundle = self.analytics.find_product(query=query, n=8)
            local_answer = f"{bundle.summary}\n{format_table_for_chat(bundle.table)}"
        elif parsed.intent == "strategy":
            local_answer = self.analytics.strategy_recommendations(
                budget_small=bool(entities.get("budget_small"))
            )
        else:
            local_answer = (
                self.analytics.strategy_recommendations(budget_small=bool(entities.get("budget_small")))
                if parsed.is_data_question
                else "I can help with general questions and warehouse data questions. Ask about KPIs, products, campaigns, CSV tables, or a broader business topic."
            )

        context, sources = self.analytics.relevant_context(
            query=user_message,
            product_hint=entities.get("query"),
            dataset_hint=entities.get("dataset_query"),
        )
        sources = self._curate_sources(parsed.intent, sources, entities)
        model_prompt = self._build_model_prompt(
            user_message=user_message,
            local_answer=local_answer,
            warehouse_context=context if parsed.is_data_question else "",
        )

        if self.config.gemini_enabled:
            gemini_answer = self.gemini.generate_answer(
                system_instruction=self.system_instruction,
                user_message=model_prompt,
                history=history,
                temperature=0.2 if parsed.is_data_question else 0.45,
                max_output_tokens=700,
            )
            if gemini_answer:
                return ChatbotReply(
                    answer=gemini_answer,
                    mode="hybrid" if parsed.is_data_question else "gemini",
                    sources=sources,
                    model=self.gemini.active_model_name or self.config.gemini_model,
                )

        if not self.gemini.available and self.config.gemini_enabled:
            local_answer = f"{local_answer}\n\nGemini note: {self.gemini.status_message()}"

        return ChatbotReply(
            answer=local_answer,
            mode="local",
            sources=sources,
            model="local",
        )

    @staticmethod
    def _compact_context(text: str, max_chars: int = 1800) -> str:
        normalized = compact_whitespace(text)
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3] + "..."

    def _build_model_prompt(self, user_message: str, local_answer: str, warehouse_context: str) -> str:
        sections = [
            f"User question:\n{user_message}",
            (
                "Local analytics answer:\n"
                f"{self._compact_context(local_answer, max_chars=2200)}"
            ),
        ]
        if warehouse_context:
            sections.append(
                "Warehouse context to use as factual grounding:\n"
                f"{self._compact_context(warehouse_context, max_chars=5000)}"
            )
            sections.append(
                "Instructions: Keep the warehouse context authoritative for numbers, tables, and schema details. "
                "If the local analytics answer already resolves the question, refine it rather than replacing it."
            )
        else:
            sections.append(
                "Instructions: Answer naturally and directly. If the question becomes data-specific, explain that you can verify it against the warehouse."
            )
        return "\n\n".join(sections)

    @staticmethod
    def _curate_sources(intent: str, sources: list[str], entities: dict) -> list[str]:
        if intent in {"kpi_summary", "top_products", "cheap_low_sales", "strategy", "search_product"}:
            return ["product_metrics"]
        if intent == "dataset_overview" and not entities.get("dataset_query"):
            return []
        return sources
