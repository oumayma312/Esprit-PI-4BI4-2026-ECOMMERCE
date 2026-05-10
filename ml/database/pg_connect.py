from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
import psycopg
from psycopg import sql


@dataclass(frozen=True)
class DbConfig:
    dbname: str = "pi_bi"
    user: str = "postgres"
    password: str = "douraid"
    host: str = "localhost"
    port: int = 5432
    connect_timeout: int = 10
    sslmode: str = "prefer"
    schema: str = "datawarehouse"
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "DbConfig":
        enabled_raw = os.getenv("ML_USE_DATABASE", "true").strip().lower()
        enabled = enabled_raw not in {"0", "false", "no"}
        return cls(
            dbname=os.getenv("PGDATABASE", "pi_bi").strip() or "pi_bi",
            user=os.getenv("PGUSER", "postgres").strip() or "postgres",
            password=os.getenv("PGPASSWORD", "douraid"),
            host=os.getenv("PGHOST", "localhost").strip() or "localhost",
            port=int(os.getenv("PGPORT", "5432")),
            connect_timeout=int(os.getenv("PGCONNECT_TIMEOUT", "10")),
            sslmode=os.getenv("PGSSLMODE", "prefer").strip() or "prefer",
            schema=os.getenv("DW_SCHEMA", "datawarehouse").strip() or "datawarehouse",
            enabled=enabled,
        )


def get_connection(config: DbConfig | None = None) -> psycopg.Connection:
    effective_config = config or DbConfig.from_env()
    connection = psycopg.connect(
        dbname=effective_config.dbname,
        user=effective_config.user,
        password=effective_config.password,
        host=effective_config.host,
        port=effective_config.port,
        connect_timeout=effective_config.connect_timeout,
        sslmode=effective_config.sslmode,
    )
    connection.autocommit = True
    return connection


def load_table(
    connection: psycopg.Connection,
    table_name: str,
    schema_name: str,
) -> pd.DataFrame:
    query = sql.SQL("SELECT * FROM {}.{};").format(
        sql.Identifier(schema_name),
        sql.Identifier(table_name),
    )
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [column.name for column in cursor.description] if cursor.description is not None else []
    return pd.DataFrame(rows, columns=columns)


def test_connection(config: DbConfig | None = None) -> tuple[bool, str]:
    effective_config = config or DbConfig.from_env()
    if not effective_config.enabled:
        return False, "Database access disabled by ML_USE_DATABASE."

    conn = None
    try:
        conn = get_connection(effective_config)
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
        return True, f"Connected to PostgreSQL: {version}"
    except psycopg.Error as exc:
        return False, str(exc)
    finally:
        if conn is not None:
            conn.close()

