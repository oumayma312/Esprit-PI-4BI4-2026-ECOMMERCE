from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.main import app as dss_app
from api.main import run_pipeline_before_api as dss_startup
from ml.api import build_ml_router

from .chatbot import MarketingChatbot
from .config import AppConfig
from .face_service import FaceRegistry, FaceServiceError, FaceValidationError

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

class FaceRegisterRequest(BaseModel):
    email: str
    role: str
    image: str
    label: str | None = None


class FaceRecognitionRequest(BaseModel):
    image: str = Field(min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    mode: str
    model: str
    sources: list[str]
    response_ms: int


config = AppConfig.from_env()
chatbot = MarketingChatbot(config)
face_registry = FaceRegistry(config.root_dir / "faces")

@asynccontextmanager
async def lifespan(_: FastAPI):
    dss_startup()
    yield


app = FastAPI(title="Story AI Chatbot API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.allowed_origins) or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(build_ml_router(config))
app.include_router(dss_app.router, prefix="/api")

sougui_assets_dir = config.root_dir / "sougui_photos"
if sougui_assets_dir.is_dir():
    app.mount("/api/assets/sougui", StaticFiles(directory=sougui_assets_dir), name="sougui-assets")


@app.get("/api/health")
def healthcheck() -> dict:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chatbot": {
            "ready": True,
            "mode": "hybrid" if config.gemini_enabled else "local",
            "model": config.gemini_model,
            "data_dir": str(config.data_dir),
            "gemini_enabled": config.gemini_enabled,
            "gemini_status": chatbot.gemini.status_message(),
        },
        "face_recognition": face_registry.health_summary(),
    }


@app.get("/api/chatbot/health")
def chatbot_healthcheck() -> dict:
    return healthcheck()


@app.post("/api/faces/register")
def register_face(payload: FaceRegisterRequest):
    try:
        return face_registry.register(
            email=payload.email,
            role=payload.role,
            image_data_url=payload.image,
            label=payload.label,
        )
    except FaceValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FaceServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/faces")
def get_faces():
    return {"profiles": face_registry.list_profiles()}


@app.post("/api/faces/recognize")
def recognize_face(payload: FaceRecognitionRequest):
    try:
        return face_registry.recognize(payload.image)
    except FaceValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FaceServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chatbot", response_model=ChatResponse)
def ask_chatbot(payload: ChatRequest) -> ChatResponse:
    try:
        started_at = perf_counter()
        reply = chatbot.respond(
            message=payload.message,
            history=[turn.model_dump() for turn in payload.history],
        )
        duration_ms = int((perf_counter() - started_at) * 1000)
        return ChatResponse(
            answer=reply.answer,
            mode=reply.mode,
            model=reply.model,
            sources=reply.sources,
            response_ms=duration_ms,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="The chatbot API timed out.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The chatbot request could not be completed.") from exc
