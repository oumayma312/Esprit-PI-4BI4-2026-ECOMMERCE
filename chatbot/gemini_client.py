from __future__ import annotations

from time import sleep
from typing import Optional

import requests


class GeminiClient:
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model_name: str, timeout_seconds: float = 25.0) -> None:
        self.available = bool(api_key.strip())
        self.api_key = api_key.strip()
        self.model_name = model_name
        self.active_model_name = ""
        self.last_error = ""
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    @staticmethod
    def _format_error(error: Exception) -> str:
        message = str(error)
        lowered = message.lower()

        if "reported as leaked" in lowered:
            return (
                "Google rejected the Gemini key because it appears to be compromised. Replace it with a new API key."
            )

        if "quota" in lowered or "resourceexhausted" in lowered:
            return (
                "The Gemini quota has been exhausted for this key or project. Enable billing or wait for the quota reset."
            )

        if "not found" in lowered:
            return "The configured Gemini model is not available for this API key. Check GEMINI_MODEL."

        return message

    def status_message(self) -> str:
        if self.available and self.active_model_name:
            return f"Gemini is active with model {self.active_model_name}."

        if self.available and not self.last_error:
            return f"Gemini is configured with model {self.model_name}."

        if self.last_error:
            return f"Gemini is unavailable: {self.last_error}"

        return "Gemini is unavailable: no model is configured."

    def generate_answer(
        self,
        system_instruction: str,
        user_message: str,
        history: list[dict] | None = None,
        temperature: float = 0.35,
        max_output_tokens: int = 700,
    ) -> Optional[str]:
        if not self.available:
            return None

        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": self._build_contents(history or [], user_message),
            "generationConfig": {
                "temperature": temperature,
                "topP": 0.9,
                "maxOutputTokens": max_output_tokens,
            },
        }

        for model_name in self._candidate_models():
            try:
                answer = self._post_generate_content(model_name, payload)
                if answer:
                    self.active_model_name = model_name
                    return answer
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                error_text = exc.response.text if exc.response is not None else str(exc)
                self.last_error = self._format_error(Exception(error_text))
                if status_code in {404, 400}:
                    continue
                if status_code in {401, 403, 429}:
                    self.available = False
                    return None
            except requests.RequestException as exc:
                self.last_error = self._format_error(exc)
                return None
            except Exception as exc:
                self.last_error = self._format_error(exc)
                return None

        self.available = False
        return None

    def _candidate_models(self) -> list[str]:
        candidates = [
            self.model_name,
            self.model_name.replace("models/", ""),
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ]
        unique: list[str] = []
        for candidate in candidates:
            normalized = candidate.strip()
            if normalized and normalized not in unique:
                unique.append(normalized)
        return unique

    @staticmethod
    def _build_contents(history: list[dict], latest_user_message: str) -> list[dict]:
        contents: list[dict] = []
        for turn in history[-6:]:
            role = "model" if turn.get("role") == "assistant" else "user"
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            contents.append({"role": role, "parts": [{"text": content}]})
        contents.append({"role": "user", "parts": [{"text": latest_user_message}]})
        return contents

    def _post_generate_content(self, model_name: str, payload: dict) -> Optional[str]:
        retry_statuses = {429, 500, 502, 503, 504}
        last_response: requests.Response | None = None

        for attempt in range(3):
            response = self.session.post(
                self.API_URL.format(model=model_name),
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout_seconds,
            )
            last_response = response

            if response.status_code in retry_statuses and attempt < 2:
                sleep(2 ** attempt)
                continue

            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return None

            parts = candidates[0].get("content", {}).get("parts", [])
            text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
            return text.strip() or None

        if last_response is not None:
            last_response.raise_for_status()
        return None
