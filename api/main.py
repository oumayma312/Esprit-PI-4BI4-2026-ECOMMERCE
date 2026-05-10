from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import pandas as pd

from pipeline.collect_kpis import compute_all_kpis, load_all_tables
from pipeline.langchain_dss import generate_role_decisions, generate_text_response
from pipeline.ml_model_artifacts import (
    export_all_model_artifacts,
    export_model_artifact,
    get_model_measures,
    list_model_artifacts,
)
from pipeline.run_ml_models import run_ml_models
from run_pipeline import run as run_pipeline_job
from pipeline.save_decisions import (
    get_decisions_by_role,
    get_decisions_summary,
    mark_decision_read,
    save_decisions,
)
from pipeline.transform_dimensions import transform_dim_customer, transform_dim_product, transform_dim_supplier
from pipeline.transform_factachat import transform_factachat
from pipeline.transform_factventee import transform_factventee

app = FastAPI(title="Decision Support System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def run_pipeline_before_api() -> None:
    # Ensure fresh decisions are generated every time the API starts.
    run_pipeline_job()


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: Optional[str] = None
    max_tokens: int = Field(800, ge=32, le=4000)


class PromptResponse(BaseModel):
    model: str
    response: str


class RoutedPromptRequest(BaseModel):
    prompt: str = Field(..., min_length=3)
    role: Optional[str] = Field(default=None, description="Optional role override: CEO/Finance/Sales")
    model: Optional[str] = None
    max_tokens: int = Field(1600, ge=64, le=4000)
    persist: bool = True


class RoutedPromptResponse(BaseModel):
    routed_role: str
    decisions: List[Dict[str, Any]]
    saved_ids: List[int] = []
    model: str


class ModelExportResponse(BaseModel):
    model_key: str
    kind: Optional[str] = None
    ok: bool
    artifact_path: Optional[str] = None
    sidecar_path: Optional[str] = None
    serializer: Optional[str] = None
    error: Optional[str] = None
    returncode: Optional[int] = None
    stdout_tail: Optional[str] = None
    stderr_tail: Optional[str] = None


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


def _coerce_numeric(dfs: Dict[str, pd.DataFrame], table: str, cols: List[str]) -> None:
    df = dfs.get(table)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return
    for c in cols:
        if c in df.columns:
            df[c] = _to_number(df[c])
    dfs[table] = df


def _route_role(user_prompt: str) -> str:
    txt = (user_prompt or "").lower()

    finance_keys = [
        "finance",
        "budget",
        "coût",
        "cout",
        "marge",
        "roi",
        "dépense",
        "depense",
        "cash",
        "trésorerie",
        "tresorerie",
    ]
    sales_keys = [
        "sales",
        "vente",
        "campagne",
        "client",
        "churn",
        "concurrent",
        "prix",
        "promotion",
        "conversion",
        "segment",
    ]

    if any(k in txt for k in finance_keys):
        return "Finance"
    if any(k in txt for k in sales_keys):
        return "Sales"
    return "CEO"


def _prepare_context_data() -> tuple[Dict[str, Any], Dict[str, Any]]:
    dfs = load_all_tables()

    if isinstance(dfs.get("FactVentee"), object) and not dfs.get("FactVentee").empty:
        raw_fv = dfs["FactVentee"].copy()
        fv_transformed = transform_factventee(raw_fv)
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

    _coerce_numeric(dfs, "FactCampaign", ["price"])
    _coerce_numeric(dfs, "FactFinance", ["total_ht"])
    _coerce_numeric(dfs, "FactAchat", ["total_ht"])
    _coerce_numeric(dfs, "FactShipping", ["deliveryPrice"])
    _coerce_numeric(dfs, "FactConcurrent", ["price_product"])
    _coerce_numeric(dfs, "FactVentee", ["total_ht", "price", "remise"])

    kpis = compute_all_kpis(dfs)
    ml_outputs = run_ml_models(dfs)
    return kpis, ml_outputs


def _compact_decision(decision: Dict[str, Any], include_snapshot: bool = False) -> Dict[str, Any]:
    compact = {
        "id": decision.get("id"),
        "date_generated": decision.get("date_generated"),
        "role": decision.get("role"),
        "titre": decision.get("titre"),
        "urgence": decision.get("urgence"),
        "sources": decision.get("sources"),
        "statut": decision.get("statut"),
        "created_at": decision.get("created_at"),
    }
    if include_snapshot:
        compact["kpis_snapshot"] = decision.get("kpis_snapshot")
    return compact


@app.get("/decisions")
def list_decisions(
    role: str = Query(..., description="Role name, e.g. CEO/Finance/Sales"),
    unread_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
    include_snapshot: bool = Query(False, description="Include heavy kpis_snapshot payload"),
) -> List[Dict[str, Any]]:
    decisions = get_decisions_by_role(role=role, unread_only=unread_only, limit=limit)
    return [_compact_decision(d, include_snapshot=include_snapshot) for d in decisions]


@app.post("/decisions/{decision_id}/read")
def mark_read(decision_id: int) -> Dict[str, Any]:
    try:
        event = mark_decision_read(decision_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"decision_id": decision_id, "statut": "lu", "event": event}


@app.get("/decisions/summary")
def summary() -> Dict[str, Any]:
    return get_decisions_summary()


@app.post("/llm/prompt", response_model=PromptResponse)
def llm_prompt(payload: PromptRequest) -> PromptResponse:
    try:
        text = generate_text_response(
            prompt=payload.prompt,
            model=payload.model,
            max_tokens=payload.max_tokens,
        )
        selected_model = payload.model or "mistral-large-latest"
        return PromptResponse(model=selected_model, response=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/route", response_model=RoutedPromptResponse)
def llm_route(payload: RoutedPromptRequest) -> RoutedPromptResponse:
    try:
        selected_role = (payload.role or _route_role(payload.prompt)).strip()
        if selected_role not in {"CEO", "Finance", "Sales"}:
            raise HTTPException(status_code=400, detail="role must be one of CEO, Finance, Sales")

        kpis, ml_outputs = _prepare_context_data()
        batch = generate_role_decisions(
            selected_role,
            kpis=kpis,
            ml_outputs=ml_outputs,
            user_prompt=payload.prompt,
            model=payload.model,
            max_tokens=payload.max_tokens,
        )
        llm_payload = batch.model_dump()
        decisions = llm_payload.get("decisions") or []

        saved_ids: List[int] = []
        if payload.persist:
            saved_ids = save_decisions(selected_role, llm_payload, kpis=kpis, ml_outputs=ml_outputs)

        return RoutedPromptResponse(
            routed_role=selected_role,
            decisions=decisions,
            saved_ids=saved_ids,
            model=payload.model or "mistral-large-latest",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/models")
def ml_models_list() -> Dict[str, Any]:
    return {"models": list_model_artifacts()}


@app.post("/ml/models/export", response_model=ModelExportResponse)
def ml_models_export(
    model_key: str = Query(..., description="Model key from /ml/models"),
) -> ModelExportResponse:
    try:
        res = export_model_artifact(model_key)
        return ModelExportResponse(**res)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ml/models/export-all")
def ml_models_export_all() -> Dict[str, Any]:
    return {"results": export_all_model_artifacts()}


@app.get("/ml/models/{model_key}/measures")
def ml_model_measures(model_key: str) -> Dict[str, Any]:
    try:
        measures = get_model_measures(model_key)
        return {"model_key": model_key, "measures": measures}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/models/{model_key}/measure")
def ml_model_measure_value(
    model_key: str,
    name: str = Query(..., description="Measure name, e.g. test_accuracy, rows"),
) -> Dict[str, Any]:
    try:
        measures = get_model_measures(model_key)
        if name not in measures:
            raise HTTPException(status_code=404, detail=f"Measure '{name}' not found for model '{model_key}'")
        return {"model_key": model_key, "name": name, "value": measures.get(name)}
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
