# -*- coding: utf-8 -*-
from sqlalchemy import create_engine

# Prefer psycopg (psycopg3) when available; fall back to psycopg2.
try:
    import psycopg as _psycopg3  # type: ignore
    _DRIVER = "psycopg"
except Exception:
    import psycopg2 as _psycopg2  # type: ignore
    _DRIVER = "psycopg2"

DB_CONFIG = {
    "host":     "localhost",
    "database": "pi_bi",
    "user":     "postgres",
    "password": "douraid",
    "port":     5432
}

def get_engine():
    url = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(url)

def get_connection():
    """Return a DB connection using preferred driver.

    This tries psycopg (psycopg3) first, then psycopg2. If psycopg2
    raises a UnicodeDecodeError, re-raise with additional context.
    """
    if _DRIVER == "psycopg":
        return _psycopg3.connect(
            dbname=DB_CONFIG.get("database"),
            user=DB_CONFIG.get("user"),
            password=DB_CONFIG.get("password"),
            host=DB_CONFIG.get("host"),
            port=DB_CONFIG.get("port"),
        )

    try:
        return _psycopg2.connect(**DB_CONFIG)
    except UnicodeDecodeError as exc:
        keys = ",".join(sorted(DB_CONFIG.keys()))
        raise UnicodeDecodeError(
            exc.encoding or "utf-8",
            exc.object,
            exc.start,
            exc.end,
            f"UnicodeDecodeError connecting with psycopg2. DB_CONFIG keys: {keys}"
        )

def create_decisions_table():
    """Create the decisions_log table if it does not yet exist."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS decision.decisions_log (
            id              SERIAL PRIMARY KEY,
            date_generated  DATE          NOT NULL DEFAULT CURRENT_DATE,
            role            VARCHAR(20)   NOT NULL,
            titre           TEXT          NOT NULL,
            texte_decision  TEXT          NOT NULL,
            pourquoi        TEXT,
            urgence         VARCHAR(10)   NOT NULL DEFAULT 'STABLE',
            sources         TEXT[],
            kpis_snapshot   JSONB,
            statut          VARCHAR(10)   NOT NULL DEFAULT 'nouveau',
            created_at      TIMESTAMP     DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Table decisions_log ready.")