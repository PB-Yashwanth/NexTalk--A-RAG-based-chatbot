"""
FastAPI Application
Main API entry point for NexTalk.
"""

import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import (
    CORS_ORIGINS,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    ALLOWED_EXTENSIONS,
    ENVIRONMENT,
    OLLAMA_MODEL,
    CHROMA_COLLECTION,
    RATE_LIMIT_CHAT,
    RATE_LIMIT_INGEST,
    API_KEY,
)
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    SourceChunk,
    HealthResponse,
    IngestResponse,
)
from app.rag.pipeline import get_answer
from app.rag.llm_engine import check_ollama_status
from app.rag.vector_store import get_vector_store
from app.rag.memory_manager import init_db, clear_history
from app.rag.ingestion import extract_text_from_upload, build_chunks_for_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiter (per IP)
limiter = Limiter(key_func=get_remote_address, default_limits=[])

app = FastAPI(
    title="NexTalk API",
    description="ChatGPT-like RAG application with local LLM",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please slow down."},
    )


@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info(f"NexTalk API started in {ENVIRONMENT} mode")
    logger.info(f"Using model: {OLLAMA_MODEL}")
    logger.info(f"ChromaDB collection: {CHROMA_COLLECTION}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    if ENVIRONMENT == "production":
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred. Please try again later."},
        )
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def require_api_key(request: Request) -> None:
    """
    Simple auth placeholder:
    - If API_KEY is set in .env, require header: X-API-Key: <API_KEY>
    - If API_KEY is empty, auth is disabled (dev-friendly).
    """
    if not API_KEY:
        return
    provided = request.headers.get("X-API-Key", "").strip()
    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", message="NexTalk backend is running")


@app.get("/health")
async def detailed_health_check():
    health = {
        "api": "ok",
        "database": "unknown",
        "vector_store": "unknown",
        "llm": "unknown",
    }

    try:
        from app.rag.memory_manager import _connect
        with _connect() as con:
            con.execute("SELECT 1")
        health["database"] = "ok"
    except Exception as e:
        health["database"] = f"error: {str(e)}"

    try:
        store = get_vector_store()
        count = store.count()
        health["vector_store"] = f"ok ({count} chunks)"
    except Exception as e:
        health["vector_store"] = f"error: {str(e)}"

    try:
        ollama_status = check_ollama_status()
        health["llm"] = "ok" if ollama_status.get("model_ready") else f"model not ready: {OLLAMA_MODEL}"
    except Exception as e:
        health["llm"] = f"error: {str(e)}"

    all_ok = all(v == "ok" or str(v).startswith("ok") for v in health.values())
    return {"status": "healthy" if all_ok else "degraded", "environment": ENVIRONMENT, "services": health}


@app.get("/status")
async def system_status():
    store = get_vector_store()
    ollama_status = check_ollama_status()

    return {
        "api": "running",
        "vector_store": {"status": "ok", "chunks_count": store.count()},
        "llm": ollama_status,
    }


def validate_upload(filename: str, file_size: int) -> None:
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE_MB}MB",
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")


@app.post("/ingest", response_model=IngestResponse)
@limiter.limit(RATE_LIMIT_INGEST)
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(default="anonymous"),
    _: None = Depends(require_api_key),
):
    try:
        data = await file.read()
        validate_upload(file.filename, len(data))

        # quick extraction check (optional but useful)
        text_preview = extract_text_from_upload(file.filename, data)
        if not text_preview or len(text_preview.strip()) < 10:
            raise HTTPException(status_code=400, detail="Could not extract text from file (or file is empty).")

        # IMPORTANT: build_chunks_for_store expects `data=...`
        doc_id, chunks = build_chunks_for_store(
            filename=file.filename,
            data=data,
            user_id=user_id,
        )

        store = get_vector_store()
        store.add_chunks(chunks)

        return IngestResponse(doc_id=doc_id, filename=file.filename, chunks_added=len(chunks))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest failed: {e}", exc_info=True)
        if ENVIRONMENT == "production":
            raise HTTPException(status_code=400, detail="Failed to process document")
        raise HTTPException(status_code=400, detail=f"Ingest failed: {str(e)}")


@app.delete("/docs/{user_id}")
async def delete_user_docs(
    request: Request,
    user_id: str,
    _: None = Depends(require_api_key),
):
    store = get_vector_store()
    try:
        store.collection.delete(where={"user_id": user_id})
        return {"status": "ok", "message": f"Deleted uploaded docs for user_id={user_id}"}
    except Exception as e:
        logger.error(f"Failed to delete docs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete documents")


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMIT_CHAT)
async def chat(
    request: Request,
    payload: ChatRequest,
    _: None = Depends(require_api_key),
):
    try:
        user_id = payload.user_id or "anonymous"

        result = get_answer(
            question=payload.message,
            top_k=payload.top_k,
            user_id=user_id,
            use_reranker=False,
            mode=payload.mode,
            active_doc_ids=payload.active_doc_ids,
        )

        sources = [
            SourceChunk(
                content=s.get("content", ""),
                source=s.get("source", "Unknown"),
                score=float(s.get("score", 0.0)),
                page=s.get("page"),
                doc_id=s.get("doc_id"),
            )
            for s in result.get("sources", [])
        ]

        return ChatResponse(answer=result["answer"], sources=sources, conversation_id=user_id)

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        if ENVIRONMENT == "production":
            raise HTTPException(status_code=500, detail="Failed to process your message")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.delete("/memory/{user_id}")
async def clear_memory(
    request: Request,
    user_id: str,
    _: None = Depends(require_api_key),
):
    clear_history(user_id)
    return {"status": "ok", "message": f"Memory cleared for user {user_id}"}