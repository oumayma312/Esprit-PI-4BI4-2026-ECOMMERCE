from __future__ import annotations

import os
from typing import Any, Dict

from config.db import create_decisions_table
import pandas as pd

from pipeline.build_context import load_prompts
from pipeline.collect_kpis import load_all_tables, compute_all_kpis
from pipeline.langchain_dss import generate_role_decisions
from pipeline.run_ml_models import run_ml_models
from pipeline.save_decisions import save_decisions
from pipeline.transform_factachat import transform_factachat
from pipeline.transform_factventee import transform_factventee
from pipeline.transform_dimensions import transform_dim_customer, transform_dim_product, transform_dim_supplier


def _to_number(series: pd.Series) -> pd.Series:
    if series is None:
        return series
    s = series.astype(str)
    s = s.str.strip()
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    s = s.replace({"": None, "(null)": None, "null": None, "nan": None, "None": None})
    return pd.to_numeric(s, errors="coerce")


def _coerce_numeric(dfs: Dict[str, pd.DataFrame], table: str, cols: list[str]) -> None:
    df = dfs.get(table)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return
    for c in cols:
        if c in df.columns:
            df[c] = _to_number(df[c])
    dfs[table] = df


def run() -> Dict[str, Any]:
    create_decisions_table()

    dfs = load_all_tables()
    if isinstance(dfs.get("FactVentee"), object) and not dfs.get("FactVentee").empty:
        raw_fv = dfs["FactVentee"].copy()
        fv_transformed = transform_factventee(raw_fv)
        # Guardrail: if transformation unexpectedly drops all rows, keep raw source.
        dfs["FactVentee"] = fv_transformed if not fv_transformed.empty else raw_fv

    if isinstance(dfs.get("FactAchat"), object) and not dfs.get("FactAchat").empty:
        raw_fa = dfs["FactAchat"].copy()
        fa_transformed = transform_factachat(raw_fa)
        dfs["FactAchat"] = fa_transformed if not fa_transformed.empty else raw_fa

    if isinstance(dfs.get("DimSupplier"), object) and not dfs.get("DimSupplier").empty:
        dfs["DimSupplier"] = transform_dim_supplier(dfs["DimSupplier"])

    if isinstance(dfs.get("DimCustomer"), object) and not dfs.get("DimCustomer").empty:
        dfs["DimCustomer"] = transform_dim_customer(dfs["DimCustomer"])

    if isinstance(dfs.get("DimProduct"), object) and not dfs.get("DimProduct").empty:
        dfs["DimProduct"] = transform_dim_product(dfs["DimProduct"])

    # Coerce numeric columns needed by compute_all_kpis (keep collect_kpis.py unchanged)
    _coerce_numeric(dfs, "FactCampaign", ["price"])
    _coerce_numeric(dfs, "FactFinance", ["total_ht"])
    _coerce_numeric(dfs, "FactAchat", ["total_ht"])
    _coerce_numeric(dfs, "FactShipping", ["deliveryPrice"])
    _coerce_numeric(dfs, "FactConcurrent", ["price_product"])
    _coerce_numeric(dfs, "FactVentee", ["total_ht", "price", "remise"])

    kpis = compute_all_kpis(dfs)
    ml_outputs = run_ml_models(dfs)

    prompts = load_prompts()
    roles = list((prompts.get("roles") or {}).keys())
    if not roles:
        raise RuntimeError("No roles found in config/prompts.json")

    results: Dict[str, Any] = {"saved": {}, "roles": roles}

    skip_llm = os.getenv("SKIP_LLM", "0") in {"1", "true", "True"}

    for role in roles:
        llm_payload = generate_role_decisions(role, kpis=kpis, ml_outputs=ml_outputs, max_tokens=1600).model_dump()

        ids = save_decisions(role, llm_payload, kpis=kpis, ml_outputs=ml_outputs)
        results["saved"][role] = ids

    return results


if __name__ == "__main__":
    out = run()
    print(out)
