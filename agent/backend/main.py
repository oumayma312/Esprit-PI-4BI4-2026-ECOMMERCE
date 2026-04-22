import uuid
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import db
from .agent import get_agent
from .tools import seed_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("koudos.backend")

# Resolve project root once for ML artifacts
_ROOT = Path(__file__).resolve().parent.parent.parent
_ARTIFACTS_DIR = str(_ROOT / "artifacts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    db.init_db()
    logger.info("Seeding manufacturers and materials...")
    seed_db()
    # Preload ML pipeline so first prediction is fast
    try:
        import sys
        sys.path.insert(0, str(_ROOT))
        from ml_pipeline import get_pipeline
        _ml_pipeline = get_pipeline(_ARTIFACTS_DIR)
        logger.info("ML pipeline loaded successfully (churn + segmentation models ready)")
    except Exception as e:
        logger.warning(f"ML pipeline preload failed (predictions will work but first call may be slow): {e}")
    yield


app = FastAPI(
    title="Artisan Materials Agent",
    description="CRUD + AI agent for managing artisan materials and orders",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---

class MaterialCreate(BaseModel):
    name: str
    category: str
    unit_price: float
    unit: str
    min_order_qty: int = 1
    stock_quantity: int = 0
    manufacturer_id: int | None = None


class MaterialUpdate(BaseModel):
    name: str | None = None
    category: str | None = None
    unit_price: float | None = None
    unit: str | None = None
    min_order_qty: int | None = None
    stock_quantity: int | None = None
    manufacturer_id: int | None = None


class ManufacturerCreate(BaseModel):
    name: str
    email: str = ""
    phone: str = ""
    location: str = ""
    materials_supplied: str = ""


class ManufacturerUpdate(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    materials_supplied: str | None = None


class OrderCreate(BaseModel):
    material_id: int
    quantity: int
    notes: str = ""


class OrderStatusUpdate(BaseModel):
    status: str


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class HITLDecision(BaseModel):
    thread_id: str
    decision: str
    edited_args: dict[str, Any] | None = None


class EditDecision(BaseModel):
    thread_id: str
    tool_name: str
    edited_args: dict[str, Any]


# --- ML Request/Response Models ---

class CustomerFeatures(BaseModel):
    recency: float
    frequency: float
    monetary_total: float
    monetary_trend: float
    product_diversity: float
    channel_diversity: float
    lifetime_days: float
    purchase_velocity: float
    avg_price: float
    channel: str = ""


class BatchPredictionRequest(BaseModel):
    data: list[dict[str, Any]]


# --- Helpers ---

def _get_assistant_content(messages):
    return [m.content for m in messages if hasattr(m, "content") and m.content and getattr(m, "type", "") in ("ai", "assistant")]


# --- CRUD: Manufacturers ---

@app.get("/api/manufacturers")
def api_list_manufacturers():
    return db.list_manufacturers()


@app.post("/api/manufacturers", status_code=201)
def api_create_manufacturer(data: ManufacturerCreate):
    return db.create_manufacturer(data.name, data.email, data.phone, data.location, data.materials_supplied)


@app.get("/api/manufacturers/{mfr_id}")
def api_get_manufacturer(mfr_id: int):
    m = db.get_manufacturer(mfr_id)
    if not m:
        raise HTTPException(404, f"Manufacturer {mfr_id} not found")
    return m


@app.put("/api/manufacturers/{mfr_id}")
def api_update_manufacturer(mfr_id: int, data: ManufacturerUpdate):
    m = db.update_manufacturer(mfr_id, **data.model_dump(exclude_none=True))
    if not m:
        raise HTTPException(404, f"Manufacturer {mfr_id} not found")
    return m


@app.delete("/api/manufacturers/{mfr_id}", status_code=204)
def api_delete_manufacturer(mfr_id: int):
    if not db.delete_manufacturer(mfr_id):
        raise HTTPException(404, f"Manufacturer {mfr_id} not found")


# --- CRUD: Materials ---

@app.get("/api/materials")
def api_list_materials():
    return db.list_materials()


@app.post("/api/materials", status_code=201)
def api_create_material(data: MaterialCreate):
    return db.create_material(data.name, data.category, data.unit_price, data.unit,
                              data.min_order_qty, data.stock_quantity, data.manufacturer_id)


@app.get("/api/materials/context")
def api_materials_context(category: str = ""):
    materials = db.list_materials()
    if category:
        materials = [m for m in materials if m["category"].lower() == category.lower()]
    return [{
        "id": m["id"],
        "name": m["name"],
        "category": m["category"],
        "min_order_qty": m["min_order_qty"],
        "stock_quantity": m.get("stock_quantity", 0),
        "unit": m["unit"],
    } for m in materials]


@app.get("/api/materials/{mat_id}")
def api_get_material(mat_id: int):
    m = db.get_material(mat_id)
    if not m:
        raise HTTPException(404, f"Material {mat_id} not found")
    return m


@app.put("/api/materials/{mat_id}")
def api_update_material(mat_id: int, data: MaterialUpdate):
    m = db.update_material(mat_id, **data.model_dump(exclude_none=True))
    if not m:
        raise HTTPException(404, f"Material {mat_id} not found")
    return m


@app.delete("/api/materials/{mat_id}", status_code=204)
def api_delete_material(mat_id: int):
    if not db.delete_material(mat_id):
        raise HTTPException(404, f"Material {mat_id} not found")


# --- CRUD: Orders ---

@app.get("/api/orders")
def api_list_orders():
    return db.list_orders()


@app.post("/api/orders", status_code=201)
def api_create_order(data: OrderCreate):
    mat = db.get_material(data.material_id)
    if not mat:
        raise HTTPException(404, f"Material {data.material_id} not found")
    if data.quantity < mat["min_order_qty"]:
        raise HTTPException(400, f"Quantity below minimum ({mat['min_order_qty']}) for {mat['name']}")
    return db.create_order(data.material_id, data.quantity, data.notes)


@app.get("/api/orders/{order_id}")
def api_get_order(order_id: int):
    o = db.get_order(order_id)
    if not o:
        raise HTTPException(404, f"Order {order_id} not found")
    return o


@app.patch("/api/orders/{order_id}/status")
def api_update_order_status(order_id: int, data: OrderStatusUpdate):
    o = db.update_order_status(order_id, data.status)
    if not o:
        raise HTTPException(404, f"Order {order_id} not found")
    return o


@app.delete("/api/orders/{order_id}", status_code=204)
def api_delete_order(order_id: int):
    if not db.delete_order(order_id):
        raise HTTPException(404, f"Order {order_id} not found")


# --- Chat & Agent ---

@app.post("/api/chat")
def api_chat(req: ChatRequest):
    agent = get_agent()
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
            version="v2",
        )
    except Exception as e:
        raise HTTPException(500, f"Agent error: {e}")

    value = result.value
    messages = value.get("messages", [])

    assistant_content = _get_assistant_content(messages)
    response = assistant_content[-1] if assistant_content else "No response."

    has_interrupts = len(result.interrupts) > 0

    interrupt_data = None
    prefill = None
    if has_interrupts and result.interrupts:
        interrupt = result.interrupts[0]
        action_requests = interrupt.value.get("action_requests", [])
        if action_requests:
            interrupt_data = action_requests[0]
            # Extract prefill data from tool call arguments (support both old single-material and new multi-material format)
            args = action_requests[0].get("args", {})
            if args and "materials" not in args and "material_id" in args:
                # Old format: create_order(material_id=1, quantity=50)
                prefill = {
                    "materials": [{"material_id": args.get("material_id"), "quantity": args.get("quantity", 1)}],
                    "notes": args.get("notes", ""),
                }
            else:
                # New format: create_order(materials=[{material_id: 1, quantity: 50}])
                prefill = {
                    "materials": args.get("materials", []),
                    "notes": args.get("notes", ""),
                }

    return {
        "thread_id": thread_id,
        "response": response,
        "pending_approval": has_interrupts,
        "interrupt_data": interrupt_data,
        "prefill": prefill,
    }


@app.get("/api/chat/stream")
async def api_chat_stream(message: str, thread_id: str | None = None):
    agent = get_agent()
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': thread_id})}\n\n"

            for chunk in agent.stream(
                {"messages": [{"role": "user", "content": message}]},
                config=config,
                stream_mode=["updates", "messages"],
                version="v2",
            ):
                if chunk["type"] == "messages":
                    token, metadata = chunk["data"]
                    if token.content:
                        yield f"data: {json.dumps({'type': 'token', 'content': token.content})}\n\n"
                elif chunk["type"] == "updates":
                    if "__interrupt__" in chunk["data"]:
                        interrupt_data = chunk["data"]["__interrupt__"][0]
                        action_requests = interrupt_data.value.get("action_requests", [])
                        # Extract prefill data from tool call arguments (support both old single-material and new multi-material format)
                        prefill = None
                        if action_requests:
                            args = action_requests[0].get("args", {})
                            if args and "materials" not in args and "material_id" in args:
                                # Old format: create_order(material_id=1, quantity=50)
                                prefill = {
                                    "materials": [{"material_id": args.get("material_id"), "quantity": args.get("quantity", 1)}],
                                    "notes": args.get("notes", ""),
                                }
                            else:
                                # New format: create_order(materials=[{material_id: 1, quantity: 50}])
                                prefill = {
                                    "materials": args.get("materials", []),
                                    "notes": args.get("notes", ""),
                                }
                        yield f"data: {json.dumps({'type': 'interrupt', 'data': action_requests, 'prefill': prefill})}\n\n"

            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/chat/approve")
def api_approve(decision: HITLDecision):
    agent = get_agent()
    thread_id = decision.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    from langgraph.types import Command

    final_response = None
    for step in agent.stream(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
        stream_mode="values",
        version="v2",
    ):
        data = step.get("data", {})
        msgs = data.get("messages", [])
        assistant_msgs = _get_assistant_content(msgs)
        if assistant_msgs:
            final_response = assistant_msgs[-1]

    return {
        "thread_id": thread_id,
        "response": final_response or "Decision recorded.",
    }


@app.post("/api/chat/reject")
def api_reject(decision: HITLDecision):
    agent = get_agent()
    thread_id = decision.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    from langgraph.types import Command

    final_response = None
    for step in agent.stream(
        Command(resume={"decisions": [{"type": "reject"}]}),
        config=config,
        stream_mode="values",
        version="v2",
    ):
        data = step.get("data", {})
        msgs = data.get("messages", [])
        assistant_msgs = _get_assistant_content(msgs)
        if assistant_msgs:
            final_response = assistant_msgs[-1]

    return {
        "thread_id": thread_id,
        "response": final_response or "Decision recorded.",
    }


@app.post("/api/chat/edit")
def api_edit(decision: EditDecision):
    agent = get_agent()
    thread_id = decision.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    from langgraph.types import Command

    final_response = None
    for step in agent.stream(
        Command(resume={"decisions": [{
            "type": "edit",
            "edited_action": {
                "name": decision.tool_name,
                "args": decision.edited_args,
            }
        }]}),
        config=config,
        stream_mode="values",
        version="v2",
    ):
        data = step.get("data", {})
        msgs = data.get("messages", [])
        assistant_msgs = _get_assistant_content(msgs)
        if assistant_msgs:
            final_response = assistant_msgs[-1]

    return {
        "thread_id": thread_id,
        "response": final_response or "Decision recorded.",
    }


# --- Email ---

class EmailTestRequest(BaseModel):
    to: str
    subject: str = "Test Notification"
    body: str = "This is a test notification from Atelier Materials."


@app.get("/api/email/status")
def api_email_status():
    """Check if email service is configured and available."""
    from .email import get_email_service
    service = get_email_service()
    return {
        "enabled": service.enabled,
        "from_email": service.from_email,
        "configured": bool(service.api_key),
    }


@app.post("/api/email/test")
def api_email_test(req: EmailTestRequest):
    """Send a test email to verify email configuration."""
    from .email import get_email_service
    service = get_email_service()
    result = service.send_notification(
        to=req.to,
        subject=req.subject,
        body=req.body,
    )
    if result:
        return {"status": "sent", "email_id": result.get("id")}
    return {"status": "failed", "message": "Email service unavailable or send failed"}


# --- ML Prediction Endpoints ---

@app.post("/api/predict/churn")
def api_predict_churn(req: CustomerFeatures):
    """Predict churn probability and segment for a single customer."""
    logger.info(f"Prediction request: recency={req.recency}, frequency={req.frequency}, monetary={req.monetary_total}")
    start = time.time()
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from ml_pipeline import get_pipeline
    pipeline = get_pipeline(_ARTIFACTS_DIR)
    try:
        result = pipeline.predict_customer(req.model_dump())
        elapsed = time.time() - start
        logger.info(f"Prediction complete: churn_prob={result['churn_probability']:.4f}, segment={result['segment']}, risk={result['risk_level']} ({elapsed:.2f}s)")
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(500, f"Prediction error: {e}")


@app.post("/api/predict/churn/batch")
def api_predict_churn_batch(req: BatchPredictionRequest):
    """Predict churn for multiple customers at once."""
    logger.info(f"Batch prediction request: {len(req.data)} customers")
    start = time.time()
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    import pandas as pd
    from ml_pipeline import get_pipeline
    pipeline = get_pipeline(_ARTIFACTS_DIR)
    if not req.data:
        raise HTTPException(400, "No data provided")
    try:
        df = pd.DataFrame(req.data)
        results, issues = pipeline.batch_predict(df)
        elapsed = time.time() - start
        logger.info(f"Batch prediction complete: {len(results)} results ({elapsed:.2f}s)")
        if issues:
            logger.warning(f"Batch prediction issues: {issues}")
        records = results.to_dict("records")
        return {"predictions": records, "issues": issues}
    except Exception as e:
        raise HTTPException(500, f"Batch prediction error: {e}")


# --- Customer Endpoints ---

@app.get("/api/customers")
def api_list_customers(
    segment: int | None = None,
    risk_level: str | None = None,
    search: str = "",
    limit: int = 100,
    offset: int = 0,
):
    """List customers with churn predictions and segments."""
    customers = db.list_customers(segment=segment, risk_level=risk_level, search=search, limit=limit, offset=offset)
    return customers


@app.get("/api/customers/count")
def api_customer_count(
    segment: int | None = None,
    risk_level: str | None = None,
    search: str = "",
):
    """Get total count of customers matching filters."""
    with db.get_db() as conn:
        query = "SELECT COUNT(*) as cnt FROM customers c LEFT JOIN customer_predictions cp ON c.id = cp.customer_id"
        params: list = []
        conditions = []
        if segment is not None:
            conditions.append("cp.segment = ?")
            params.append(segment)
        if risk_level is not None:
            conditions.append("cp.risk_level = ?")
            params.append(risk_level)
        if search:
            conditions.append("c.name LIKE ?")
            params.append(f"%{search}%")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        row = conn.execute(query, params).fetchone()
        return {"count": row["cnt"]}


@app.get("/api/customers/high-risk")
def api_high_risk_customers(min_probability: float = 0.7):
    """List customers with high churn risk."""
    customers = db.get_high_risk_customers(min_probability=min_probability)
    return customers


@app.get("/api/customers/segments")
def api_segment_stats():
    """Get statistics per customer segment."""
    stats = db.get_segment_stats()
    return stats


@app.get("/api/customers/{customer_id}")
def api_get_customer(customer_id: int):
    """Get customer detail with predictions and transactions."""
    customer = db.get_customer(customer_id)
    if not customer:
        raise HTTPException(404, f"Customer {customer_id} not found")
    return customer


@app.get("/api/predict/pca-plot")
def api_pca_plot():
    """Return PCA segment visualization as PNG."""
    logger.info("Generating PCA plot...")
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from ml_pipeline import get_pipeline
    pipeline = get_pipeline(_ARTIFACTS_DIR)
    try:
        png_bytes = pipeline.generate_pca_plot()
        if not png_bytes:
            logger.warning("PCA plot not available (missing data)")
            raise HTTPException(404, "PCA plot not available")
        logger.info(f"PCA plot generated ({len(png_bytes)} bytes)")
        from fastapi.responses import Response
        return Response(content=png_bytes, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PCA plot error: {e}")
        raise HTTPException(500, f"PCA plot error: {e}")


# --- Health ---

@app.get("/api/health")
def api_health():
    return {"status": "ok", "db": "materials.db"}
