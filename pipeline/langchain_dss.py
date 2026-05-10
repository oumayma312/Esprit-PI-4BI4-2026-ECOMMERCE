from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field

from pipeline.build_context import build_role_prompt_payload, build_role_prompt_template


class DecisionItem(BaseModel):
    titre: str = Field(..., min_length=1)
    texte_decision: str = Field(..., min_length=1)
    pourquoi: str = Field(..., min_length=1)
    urgence: Literal["HAUTE", "MOYENNE", "STABLE"] = "STABLE"
    sources: List[str] = Field(default_factory=list)


class DecisionBatch(BaseModel):
    decisions: List[DecisionItem] = Field(..., min_length=1)


def _load_dotenv() -> None:
    """Load key=value pairs from project .env into the current process."""

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _build_llm(model: Optional[str] = None, max_tokens: int = 1200) -> ChatMistralAI:
    return ChatMistralAI(
        model=model or os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
        temperature=float(os.getenv("MISTRAL_TEMPERATURE", "0.2")),
        max_tokens=max_tokens,
    )


def build_decision_chain(role: str, model: Optional[str] = None, max_tokens: int = 1200):
    parser = PydanticOutputParser(pydantic_object=DecisionBatch)
    prompt = build_role_prompt_template(role).partial(format_instructions=parser.get_format_instructions())
    llm = _build_llm(model=model, max_tokens=max_tokens)

    # One-line pipeline: prompt -> LLM -> JSON parser
    return prompt | llm | parser


def generate_role_decisions(
    role: str,
    kpis: Dict[str, Any],
    ml_outputs: Dict[str, Any],
    user_prompt: str = "Analyse les données disponibles et génère les décisions pour ce rôle.",
    model: Optional[str] = None,
    max_tokens: int = 1600,
) -> DecisionBatch:
    parser = PydanticOutputParser(pydantic_object=DecisionBatch)
    prompt = build_role_prompt_template(role).partial(format_instructions=parser.get_format_instructions())
    llm = _build_llm(model=model, max_tokens=max_tokens)
    chain = prompt | llm | parser

    payload = build_role_prompt_payload(role, kpis, ml_outputs, user_prompt=user_prompt)
    payload["format_instructions"] = parser.get_format_instructions()

    result = chain.invoke(payload)

    expected_count = int(payload.get("nb_decisions") or 0)
    if expected_count > 0 and len(result.decisions) != expected_count:
        raise ValueError(
            f"LangChain output has {len(result.decisions)} decisions, expected {expected_count} for role {role}"
        )

    return result


def generate_text_response(prompt: str, model: Optional[str] = None, max_tokens: int = 800) -> str:
    llm = _build_llm(model=model, max_tokens=max_tokens)
    chain = llm | StrOutputParser()
    return chain.invoke(prompt)
