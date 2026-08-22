"""HTTP API. POST /chat is the one that matters; /ui serves the chat page.

  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from rag.config import get_settings
from rag.conversation import MAX_HISTORY_TURNS, Turn
from rag.pipeline import RateLimited, get_pipeline

settings = get_settings()


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(
        default=None, ge=1, le=20, description="Override the default top-k retrieval."
    )
    history: List[HistoryTurn] = Field(
        default_factory=list,
        max_length=40,
        description=(
            "Prior turns, oldest first. Lets a follow-up like \"who teaches it?\" "
            "resolve against what was asked before. Only the last "
            f"{MAX_HISTORY_TURNS} are used."
        ),
    )


class SourceOut(BaseModel):
    citation: int
    source_name: str
    department: str
    url: str
    page: Optional[int] = None
    heading: str = ""
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceOut]
    latency_ms: int
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and open Qdrant at startup, not on the first request.
    get_pipeline()
    yield
    get_pipeline().close()


app = FastAPI(
    title="AskIITK",
    version="1.0.0",
    description=(
        "RAG over a hand-picked set of official IIT Kanpur sources, with citations."
    ),
    lifespan=lifespan,
)


UI_PATH = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@app.get("/ui", include_in_schema=False)
def ui() -> FileResponse:
    """One static page that talks to /chat."""
    return FileResponse(UI_PATH, media_type="text/html")


@app.get("/health")
def health() -> dict:
    stats = get_pipeline().stats()
    return {
        "status": "ok" if stats["points"] > 0 else "empty_index",
        "qdrant": stats,
        "embed_model": settings.embed_model,
        "llm": settings.gemini_model,
        "generation_enabled": bool(settings.gemini_api_key),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    pipeline = get_pipeline()
    try:
        result = pipeline.answer(
            request.question,
            top_k=request.top_k,
            history=[Turn(role=t.role, content=t.content) for t in request.history],
        )
    except RateLimited as exc:
        # Retry-After so the caller can back off instead of guessing.
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    except RuntimeError as exc:  # missing API key: config, not a bug
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=repr(exc)) from exc

    return ChatResponse(
        answer=result.answer,
        sources=[SourceOut(**s) for s in result.sources],
        latency_ms=result.latency_ms,
        model=result.model,
    )


@app.post("/retrieve")
def retrieve(request: ChatRequest) -> dict:
    """Retrieval only, no LLM call. Used by the eval script and for debugging."""
    pipeline = get_pipeline()
    passages = pipeline.retrieve(
        request.question,
        top_k=request.top_k,
        history=[Turn(role=t.role, content=t.content) for t in request.history],
    )
    return {"question": request.question, "passages": [p.as_dict() for p in passages]}
