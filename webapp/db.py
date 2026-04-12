import os
from functools import lru_cache
from urllib.parse import quote_plus

from sqlalchemy import create_engine


def _env(name: str, fallback: str | None = None) -> str | None:
    val = os.getenv(name)
    if val is not None and str(val).strip() == "":
        return None
    return val if val is not None else fallback


@lru_cache(maxsize=1)
def get_engine():
    host = _env("PGHOST") or _env("DB_HOST") or "localhost"
    port = int((_env("PGPORT") or _env("DB_PORT") or "5432").strip())
    database = _env("PGDATABASE") or _env("DB_NAME") or "pi_bi"
    user = _env("PGUSER") or _env("DB_USER") or "postgres"
    password = _env("PGPASSWORD") or _env("DB_PASSWORD") or "04062003"

    url = f"postgresql+psycopg2://{user}:{quote_plus(password)}@{host}:{port}/{database}"
    return create_engine(url, pool_pre_ping=True)
