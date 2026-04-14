import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    data_dir: str = os.getenv("DATA_DIR", "./datawarehouse")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key.strip())
