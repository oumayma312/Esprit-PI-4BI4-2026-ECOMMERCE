from sqlalchemy import create_engine
import psycopg2

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
    return psycopg2.connect(**DB_CONFIG)

def create_decisions_table():
    """Crée la table decisions_log si elle n'existe pas encore"""
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
    print("Table decisions_log prête.")