"""Connexion PostgreSQL 15 avec gestion d'erreurs.

Prerequis:
- PostgreSQL 15 en cours d'execution
- Base: pi_bi
- Utilisateur: postgres
- Mot de passe: douraid
"""

from __future__ import annotations

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


def get_connection(config: DbConfig | None = None) -> psycopg.Connection:
    """Ouvre et retourne une connexion PostgreSQL parametrable."""
    effective_config = config or DbConfig()
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


def load_table(connection: psycopg.Connection, table_name: str, schema_name: str = "public") -> pd.DataFrame:
    """Charge une table PostgreSQL dans un DataFrame pandas en securisant l'identifiant."""
    query = sql.SQL("SELECT * FROM {}.{};").format(sql.Identifier(schema_name), sql.Identifier(table_name))
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [column.name for column in cursor.description] if cursor.description is not None else []
    return pd.DataFrame(rows, columns=columns)


def test_connection(config: DbConfig) -> bool:
    """Tente une connexion PostgreSQL et retourne True si succes."""
    conn = None
    try:
        conn = psycopg.connect(
            dbname=config.dbname,
            user=config.user,
            password=config.password,
            host=config.host,
            port=config.port,
            connect_timeout=config.connect_timeout,
            sslmode=config.sslmode,
        )

        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            server_version = cur.fetchone()[0]
            cur.execute("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid();")
            ssl_row = cur.fetchone()
            ssl_enabled = bool(ssl_row[0]) if ssl_row is not None else False

        print("Connexion reussie a PostgreSQL.")
        if ssl_enabled:
            print("Canal securise SSL/TLS actif.")
        else:
            print("Attention: connexion sans SSL (serveur local sans TLS).")
        print(f"Version du serveur: {server_version}")
        return True

    except psycopg.OperationalError as err:
        print("Echec de connexion a PostgreSQL.")
        print(f"Details: {err}")
        return False

    finally:
        if conn is not None:
            conn.close()


def main() -> int:
    config = DbConfig()
    success = test_connection(config)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
