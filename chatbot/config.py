import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    data_dir: Path
    gemini_api_key: str
    gemini_model: str
    request_timeout_seconds: float
    api_host: str
    api_port: int
    allowed_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "AppConfig":
        data_dir_raw = os.getenv("DATA_DIR", "./datawarehouse")
        data_dir = Path(data_dir_raw)
        if not data_dir.is_absolute():
            data_dir = (ROOT_DIR / data_dir).resolve()

        allowed_origins = tuple(
            origin.strip()
            for origin in os.getenv(
                "CHATBOT_ALLOWED_ORIGINS",
                "http://localhost:4200,http://127.0.0.1:4200",
            ).split(",")
            if origin.strip()
        )

        return cls(
            root_dir=ROOT_DIR,
            data_dir=data_dir,
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip(),
            request_timeout_seconds=float(os.getenv("GEMINI_REQUEST_TIMEOUT", "25")),
            api_host=os.getenv("CHATBOT_API_HOST", "127.0.0.1").strip() or "127.0.0.1",
            api_port=int(os.getenv("CHATBOT_API_PORT", "8000")),
            allowed_origins=allowed_origins,
        )

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.gemini_api_key)
