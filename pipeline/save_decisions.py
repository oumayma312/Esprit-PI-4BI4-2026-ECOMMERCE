from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from config.db import create_decisions_table, get_engine


def create_decisions_read_table() -> None:
    """Create a read-log table to track read events without updating decisions_log."""

    engine = get_engine()
    from sqlalchemy import text

    sql = """
    CREATE TABLE IF NOT EXISTS decision.decisions_read_log (
        id BIGSERIAL PRIMARY KEY,
        decision_id BIGINT NOT NULL,
        read_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_decisions_read_log_decision_id
        ON decision.decisions_read_log(decision_id);
    """

    with engine.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))


def _json_default(o: Any):
    # numpy / pandas scalar -> python
    try:
        import numpy as np

        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass

    # pandas Timestamp, etc.
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            pass

    return str(o)


def _normalize_decisions(payload: Dict[str, Any], default_role: str) -> List[Dict[str, Any]]:
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("LLM JSON must contain a 'decisions' list")

    out: List[Dict[str, Any]] = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        role = str(d.get("role") or default_role)[:20]

        # New schema (FR) preferred
        titre = d.get("titre")
        texte_decision = d.get("texte_decision")
        pourquoi = d.get("pourquoi")
        urgence = d.get("urgence")
        sources = d.get("sources")

        # Backward-compatible schema (EN)
        if titre is None:
            titre = d.get("title")
        if pourquoi is None:
            pourquoi = d.get("rationale")

        actions = d.get("actions")
        if texte_decision is None and actions is not None:
            if isinstance(actions, list):
                texte_decision = "\n".join([f"- {a}" for a in actions])
            else:
                texte_decision = str(actions)

        # Urgence mapping if only priority provided
        if urgence is None:
            try:
                priority = int(d.get("priority", 3))
            except Exception:
                priority = 3
            if priority <= 2:
                urgence = "FAIBLE"
            elif priority == 3:
                urgence = "STABLE"
            else:
                urgence = "URGENT"

        titre = str(titre or "(untitled)").strip() or "(untitled)"
        texte_decision = str(texte_decision or "").strip()
        pourquoi = str(pourquoi or "").strip()

        if isinstance(sources, list):
            sources = [str(x) for x in sources]
        elif sources is None:
            sources = None
        else:
            sources = [str(sources)]

        out.append(
            {
                "role": role,
                "titre": titre,
                "texte_decision": texte_decision,
                "pourquoi": pourquoi,
                "urgence": urgence,
                "sources": sources,
                "kpis_snapshot": None,
                "statut": "nouveau",
            }
        )

    return out


def save_decisions(
    role: str,
    llm_payload: Dict[str, Any],
    kpis: Optional[Dict[str, Any]] = None,
    ml_outputs: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """Insert decisions into decision.decisions_log and return inserted ids."""

    create_decisions_table()
    engine = get_engine()

    decisions = _normalize_decisions(llm_payload, default_role=role)

    # Store snapshots as JSONB (kpis_snapshot) and include ML in the same snapshot
    snapshot = {"kpis": kpis or {}, "ml": ml_outputs or {}}

    rows = []
    for d in decisions:
        rows.append({**d, "kpis_snapshot": json.dumps(snapshot, ensure_ascii=False, default=_json_default)})

    ids: List[int] = []
    from sqlalchemy import text

    sql = """
    INSERT INTO decision.decisions_log (role, titre, texte_decision, pourquoi, urgence, sources, kpis_snapshot, statut)
    VALUES (:role, :titre, :texte_decision, :pourquoi, :urgence, :sources, CAST(:kpis_snapshot AS jsonb), :statut)
    RETURNING id
    """

    with engine.begin() as conn:
        for r in rows:
            res = conn.execute(text(sql), r)
            ids.append(int(res.scalar_one()))

    return ids


def get_decisions_by_role(role: str, unread_only: bool = False, limit: int = 50) -> List[Dict[str, Any]]:
    create_decisions_table()
    create_decisions_read_table()
    engine = get_engine()

    from sqlalchemy import text

    sql = """
    SELECT d.id,
           d.date_generated,
           d.role,
           d.titre,
           d.texte_decision,
           d.pourquoi,
           d.urgence,
           d.sources,
           d.kpis_snapshot,
           CASE WHEN r.decision_id IS NULL THEN 'nouveau' ELSE 'lu' END AS statut,
           d.created_at
    FROM decision.decisions_log d
    LEFT JOIN (
        SELECT DISTINCT decision_id
        FROM decision.decisions_read_log
    ) r
      ON r.decision_id = d.id
    WHERE d.role = :role
      AND (:unread_only = false OR r.decision_id IS NULL)
    ORDER BY d.created_at DESC
    LIMIT :limit
    """

    with engine.connect() as conn:
        res = conn.execute(text(sql), {"role": role, "unread_only": unread_only, "limit": limit})
        return [dict(r._mapping) for r in res]


def mark_decision_read(decision_id: int) -> Dict[str, Any]:
    create_decisions_table()
    create_decisions_read_table()
    engine = get_engine()

    from sqlalchemy import text

    with engine.begin() as conn:
        res = conn.execute(
            text(
                """
                INSERT INTO decision.decisions_read_log (decision_id)
                VALUES (:id)
                RETURNING id, decision_id, read_at
                """
            ),
            {"id": decision_id},
        )
        row = res.mappings().first()
        return dict(row) if row else {"decision_id": decision_id}


def get_decisions_summary() -> Dict[str, Any]:
    create_decisions_table()
    create_decisions_read_table()
    engine = get_engine()

    from sqlalchemy import text

    sql = """
    SELECT d.role,
           COUNT(*) AS total,
           SUM(CASE WHEN r.decision_id IS NULL THEN 1 ELSE 0 END) AS unread
    FROM decision.decisions_log d
    LEFT JOIN (
        SELECT DISTINCT decision_id
        FROM decision.decisions_read_log
    ) r
      ON r.decision_id = d.id
    GROUP BY d.role
    ORDER BY d.role
    """

    with engine.connect() as conn:
        res = conn.execute(text(sql))
        rows = [dict(r._mapping) for r in res]

    return {"by_role": rows}
