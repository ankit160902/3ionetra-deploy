"""
Main FastAPI application for 3ioNetra Spiritual Companion
Lean bootstrap script that orchestrates modular routers and heavy component initialization.
"""
import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request

from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import settings

# Configurable thread pool — default 64; tune via THREAD_POOL_SIZE env var
asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor(max_workers=settings.THREAD_POOL_SIZE))
from rag.pipeline import RAGPipeline

# Import routers after they are created
from routers import auth, chat, admin, memory
from routers.dependencies import set_rag_pipeline

# Setup logging via dictConfig — see logging_config.py for the rationale.
# This must run before the first logger.info() call so every log line
# (including ones from uvicorn child loggers) goes through the correlation
# filter and uses the format string that requires the correlation_id field.
from logging_config import configure as configure_logging
configure_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI tasks (Startup/Shutdown)"""
    logger.info("Starting 3ioNetra Spiritual Companion API Refinement Phase...")
    
    # 1. Initialize RAG Pipeline
    logger.info("Initializing RAG Pipeline (High Precision)...")
    rag_pipe = RAGPipeline()
    await rag_pipe.initialize()
    
    # 2. Inject RAG pipeline into routers (single shared reference)
    set_rag_pipeline(rag_pipe)
    
    # 3. Initialize other services
    from services.companion_engine import get_companion_engine
    companion_engine = get_companion_engine()
    companion_engine.set_rag_pipeline(rag_pipe)

    # 3a. Dynamic memory system (spec 2026-04-12-dynamic-memory-design.md).
    # Eagerly initialize the LongTermMemoryService singleton so the MongoDB
    # index creation in LongTermMemoryService._ensure_indexes() runs at boot
    # rather than on the first request. The index set includes the new
    # bi-temporal indexes on user_memories
    # ((user_id, invalid_at), (user_id, importance -1), (user_id, sensitivity))
    # plus the user_profiles unique index — all required by the new reader
    # and the /api/memory router. CompanionEngine already triggers this
    # transitively via its __init__, but we call it again explicitly so a
    # future refactor of engine.__init__ can't silently break index creation.
    from services.memory_service import get_memory_service
    get_memory_service(rag_pipe)
    # Import the dynamic-memory modules so any syntax or circular-import
    # error surfaces at boot rather than on the first request that uses
    # them. All four are free-function modules with no instantiation cost.
    from services import memory_reader  # noqa: F401
    from services import memory_writer  # noqa: F401
    from services import memory_extractor  # noqa: F401
    from services import reflection_service  # noqa: F401
    from services import crisis_memory_hook  # noqa: F401
    logger.info(
        "Dynamic memory system ready "
        "(reader + writer + extractor + reflection + crisis hook, "
        "indexes ensured)"
    )

    # 3b. Pre-initialize cache (surface Redis issues at boot, not first request)
    from services.cache_service import get_cache_service
    get_cache_service()

    # 3c. Pre-initialize ConceptOntology (loads SCO graph, avoids first-request latency)
    from services.concept_ontology import get_concept_ontology
    get_concept_ontology()

    # 4. Initialize query logger (async SQLite for RAG analytics)
    if settings.QUERY_LOG_ENABLED:
        from services.query_logger import get_query_logger
        await get_query_logger().initialize()

    # 5. Pre-warm Gemini context caches (avoids 30-35s penalty on first request)
    from llm.service import get_llm_service
    _llm = get_llm_service()
    if _llm.available:
        logger.info("Pre-warming Gemini context caches...")
        _llm.prewarm_caches()
        logger.info("✅ Gemini caches pre-warmed")

    logger.info("🎉 3ioNetra Backend Successfully Initialized!")

    yield
    
    # Shutdown logic
    logger.info("Shutting down 3ioNetra Spiritual Companion API...")
    
    # 1. Close Cache Connections + shared Redis pool
    from services.cache_service import get_cache_service
    cache_service = get_cache_service()
    await cache_service.close()
    from services.redis_pool import close_redis_pool
    await close_redis_pool()
    
    # 2. Close MongoDB Connections
    from services.auth_service import close_mongo_client, close_motor_client
    close_mongo_client()
    close_motor_client()

    # 3. Close query logger
    if settings.QUERY_LOG_ENABLED:
        from services.query_logger import get_query_logger
        await get_query_logger().close()

    # 4. Shut down ONNX reranker worker process pool (if active)
    try:
        from rag.onnx_reranker import _process_pool
        if _process_pool is not None:
            _process_pool.shutdown(wait=False)
            logger.info("ONNX reranker worker pool shut down")
    except Exception:
        pass

    logger.info("👋 3ioNetra Backend Shutdown Complete.")

# Initialize FastAPI app
app = FastAPI(
    title="3ioNetra Spiritual Companion API",
    version=settings.API_VERSION,
    description="3ioNetra - A modularized, high-performance spiritual bot backend.",
    lifespan=lifespan
)

# Request latency middleware — pure ASGI (does NOT buffer response bodies, preserving SSE streaming)
# NOTE: BaseHTTPMiddleware buffers the entire response before sending, which breaks StreamingResponse.
# IMPORTANT: The early-return below for non-`http` scopes (e.g. ASGI lifespan) is required;
# do not remove it or the app will fail to start.
class LatencyMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                elapsed_ms = round((time.perf_counter() - start) * 1000)
                headers = list(message.get("headers", []))
                headers.append((b"x-response-time-ms", str(elapsed_ms).encode()))
                message = {**message, "headers": headers}

                path = scope.get("path", "")
                method = scope.get("method", "")
                status_code = message.get("status", 0)
                if path.startswith("/api") and path not in ("/api/health", "/api/ready"):
                    logger.info(f"REQ_LATENCY {method} {path} {status_code} {elapsed_ms}ms")
            await send(message)

        await self.app(scope, receive, send_wrapper)

app.add_middleware(LatencyMiddleware)

# CORS middleware
_default_origins = [
    "https://ionetra-frontend-688398835360.asia-south1.run.app",
    "https://ionetra-frontend-snh4yqlhmq-el.a.run.app",
    "https://3io-netra.vercel.app",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:8000",
    "http://localhost:8080",
]
_extra = os.environ.get("ALLOWED_ORIGINS", "")
_extra_origins = [o.strip() for o in _extra.split(",") if o.strip()] if _extra else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(admin.router)
app.include_router(memory.router)

@app.get("/")
async def root():
    return {
        "app": "3ioNetra API",
        "version": settings.API_VERSION,
        "mode": "modular_refined"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
