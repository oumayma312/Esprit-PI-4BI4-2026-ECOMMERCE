from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate


ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _read_json(path: str) -> Dict[str, Any]:
    return json.loads(_resolve_path(path).read_text(encoding="utf-8"))


def load_thresholds(path: str = "config/thresholds.json") -> Dict[str, Any]:
    return _read_json(path)


def load_prompts(path: str = "config/prompts.json") -> Dict[str, Any]:
    return _read_json(path)


def _to_pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def build_role_prompt_payload(
    role: str,
    kpis: Dict[str, Any],
    ml_outputs: Dict[str, Any],
    user_prompt: str = "Analyse les données disponibles et génère les décisions pour ce rôle.",
) -> Dict[str, Any]:
    prompts = load_prompts()
    thresholds = load_thresholds()

    role_cfg = prompts.get("roles", {}).get(role)
    if not role_cfg:
        raise ValueError(f"Unknown role {role!r} in config/prompts.json")

    # Support both formats:
    # 1) {"roles": {"CEO": {...}}}
    # 2) {"CEO": {...}, "Finance": {...}, "global": {...}}
    role_thresholds = thresholds.get("roles", {}).get(role)
    if role_thresholds is None:
        role_thresholds = thresholds.get(role, {})

    # Thresholds may contain per-role thresholds + global defaults
    global_thresholds = thresholds.get("global", {}) if isinstance(thresholds, dict) else {}

    base_system_prompt = str(prompts.get("system_prompt", "")).strip()
    role_system_prompt = str(role_cfg.get("system_prompt", "")).strip()
    system_prompt = "\n\n".join([s for s in [base_system_prompt, role_system_prompt] if s]).strip()

    output_format = role_cfg.get("output_format") or prompts.get("output_format", {})
    role_desc = str(role_cfg.get("description", "")).strip()
    role_focus = str(role_cfg.get("focus", "")).strip()
    role_instructions = str(role_cfg.get("instructions", "")).strip()
    nb_decisions = int(role_cfg.get("nb_decisions") or 0)

    nb_line = f"Nb decisions attendues: {nb_decisions}"
    if nb_decisions > 0:
        nb_line = f"Nb decisions attendues: {nb_decisions} (EXACTEMENT {nb_decisions} éléments dans decisions)"

    return {
        "system_prompt": system_prompt,
        "role": role,
        "role_description": role_desc,
        "role_focus": role_focus,
        "nb_decisions": nb_decisions,
        "nb_line": nb_line,
        "role_instructions": role_instructions,
        "thresholds_role": _to_pretty_json(role_thresholds),
        "thresholds_global": _to_pretty_json(global_thresholds),
        "kpis": _to_pretty_json(kpis),
        "ml_outputs": _to_pretty_json(ml_outputs),
        "output_format": _to_pretty_json(output_format),
        "user_prompt": user_prompt,
    }


def build_role_prompt_template(role: str) -> ChatPromptTemplate:
    system_template = (
        "{system_prompt}\n\n"
        "ROLE={role}\n"
        "Description: {role_description}\n"
        "Focus: {role_focus}\n"
        "{nb_line}\n\n"
        "Instructions role (à respecter):\n{role_instructions}\n\n"
        "Thresholds (role):\n{thresholds_role}\n\n"
        "Thresholds (global):\n{thresholds_global}\n\n"
        "KPIs:\n{kpis}\n\n"
        "ML outputs:\n{ml_outputs}\n\n"
        "Output format attendu:\n{output_format}\n\n"
        "{format_instructions}\n\n"
        "Contraintes:\n"
        "- Réponds UNIQUEMENT avec un JSON valide (sans texte autour, sans Markdown).\n"
        "- Le JSON DOIT contenir la clé 'decisions' (liste).\n"
        "- Ne renvoie aucune donnée non présente dans KPIs/ML/Thresholds (pas d'invention)."
    )

    human_template = "Demande utilisateur:\n{user_prompt}"

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", human_template),
        ]
    )


def render_role_context(role: str, kpis: Dict[str, Any], ml_outputs: Dict[str, Any], user_prompt: str = "") -> str:
    prompt = build_role_prompt_template(role)
    payload = build_role_prompt_payload(role, kpis, ml_outputs, user_prompt=user_prompt)
    payload["format_instructions"] = ""
    messages = prompt.format_messages(**payload)
    return "\n\n".join(str(message.content) for message in messages).strip()


def build_context(role: str, kpis: Dict[str, Any], ml_outputs: Dict[str, Any]) -> str:
    return render_role_context(role, kpis, ml_outputs)
