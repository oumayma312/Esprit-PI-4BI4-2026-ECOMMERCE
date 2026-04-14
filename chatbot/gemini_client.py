from __future__ import annotations

from typing import Optional

import google.generativeai as genai


class GeminiClient:
    def __init__(self, api_key: str, model_name: str) -> None:
        self.available = bool(api_key.strip())
        self.model_name = model_name
        self.active_model_name = ""
        self.model = None

        if self.available:
            genai.configure(api_key=api_key)
            self.model = self._build_model(model_name)

    def _build_model(self, preferred_model: str):
        candidates = [
            preferred_model,
            preferred_model.replace("models/", ""),
            f"models/{preferred_model}",
            "models/gemini-2.0-flash",
            "models/gemini-2.5-flash",
            "models/gemini-flash-latest",
        ]

        tried = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized or normalized in tried:
                continue
            tried.add(normalized)

            try:
                model = genai.GenerativeModel(normalized)
                _ = model.generate_content("ping")
                self.active_model_name = normalized
                return model
            except Exception:
                continue

        self.available = False
        return None

    def generate_initial_answer(self, user_message: str) -> Optional[str]:
        if not self.available or self.model is None:
            return None

        prompt = (
            "Tu es un assistant marketing francophone. "
            "Donne une premiere reponse naturelle, utile et concise. "
            "N'invente pas de chiffres ni de faits non verifies. "
            "Si une precision data est necessaire, mentionne que tu vas verifier les donnees.\n\n"
            f"Question utilisateur: {user_message}\n\n"
            "Reponse en 4-8 phrases maximum."
        )

        try:
            response = self.model.generate_content(prompt)
            if response and getattr(response, "text", None):
                return response.text.strip()
        except Exception:
            return None

        return None

    def compose_hybrid_answer(
        self,
        user_message: str,
        gemini_draft: str,
        csv_context: str,
        prioritize_csv: bool,
    ) -> Optional[str]:
        if not self.available or self.model is None:
            return None

        priority_line = (
            "La question est liee aux CSV: priorise strictement les donnees CSV pour les faits et chiffres."
            if prioritize_csv
            else "Combine la reponse Gemini et le contexte CSV de facon equilibree."
        )

        prompt = (
            "Tu es un assistant marketing francophone. "
            "Produis une reponse fluide, coherente et actionnable. "
            "Si les donnees CSV contredisent le brouillon, garde les CSV. "
            "N'invente aucune metrique.\n\n"
            f"Question utilisateur: {user_message}\n\n"
            f"Brouillon Gemini:\n{gemini_draft}\n\n"
            f"Contexte CSV analyse localement:\n{csv_context}\n\n"
            f"Regle de priorite: {priority_line}\n\n"
            "Reponse finale en 6-10 phrases maximum, ton naturel et professionnel."
        )

        try:
            response = self.model.generate_content(prompt)
            if response and getattr(response, "text", None):
                return response.text.strip()
        except Exception:
            return None

        return None
