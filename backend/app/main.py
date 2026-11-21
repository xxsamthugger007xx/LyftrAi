import asyncio
import logging

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .config import Settings
from .db import Database
from .llm import AllProvidersUnavailable, LLMGateway, parse_simulation

logging.basicConfig(level=logging.INFO)
for _n in ("httpx", "LiteLLM", "litellm"):
    logging.getLogger(_n).setLevel(logging.WARNING)  # httpx logs URLs that contain the API key


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str
    system_prompt: str | None = None

    @field_validator("message")
    @classmethod
    def _message(cls, v: str) -> str:
        v = v.strip()
        if not 1 <= len(v) <= 4000:
            raise ValueError("message must be 1-4000 characters")
        return v

    @field_validator("system_prompt")
    @classmethod
    def _sp(cls, v):
        v = v.strip() if v else None
        return v or None


def create_app(settings: Settings | None = None, gateway: LLMGateway | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="Lyftr Chatbot")
    db = Database(settings.db_path)
    gateway = gateway or LLMGateway(settings)
    app.state.db, app.state.gateway = db, gateway

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
        return JSONResponse(status_code=422, content={"detail": msg or "invalid request"})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logging.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "Lyftr Chat API",
            "status": "ok",
            "health": "/health",
            "docs": "/docs",
            "endpoints": ["GET /health", "POST /chat", "GET /sessions/{session_id}/messages"],
        }

    @app.get("/health")
    async def health():
        await asyncio.to_thread(db.ping)
        return {"status": "ok", "keys_configured": len(gateway.keys), "cache_enabled": settings.cache_enabled}

    @app.post("/chat")
    async def chat(req: ChatRequest, x_simulate_failure: str | None = Header(default=None)):
        sid = req.session_id
        stored_prompt = await asyncio.to_thread(db.get_system_prompt, sid)
        system_prompt = req.system_prompt or stored_prompt or settings.default_system_prompt
        history = await asyncio.to_thread(db.history, sid)
        messages = (
            [{"role": "system", "content": system_prompt}]
            + [{"role": m["role"], "content": m["content"]} for m in history]
            + [{"role": "user", "content": req.message}]
        )
        simulate = parse_simulation(x_simulate_failure) if settings.simulation_enabled else {}
        try:
            result = await gateway.complete(messages, simulate)
        except AllProvidersUnavailable:
            raise HTTPException(status_code=503, detail="all providers unavailable")
        metadata = {
            "resolution_source": result.resolution_source,
            "model": result.model,
            "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "key_id": result.key_id,
            "fallback_attempts": result.fallback_attempts,
        }
        await asyncio.to_thread(db.add_exchange, sid, req.system_prompt, req.message, result.reply, metadata)
        return {"session_id": sid, "reply": result.reply, "metadata": metadata}

    @app.get("/sessions/{session_id}/messages")
    async def messages(session_id: str):
        history = await asyncio.to_thread(db.history, session_id)
        if not history:
            raise HTTPException(status_code=404, detail="session not found")
        return {"session_id": session_id, "messages": history}

    return app


app = create_app()
