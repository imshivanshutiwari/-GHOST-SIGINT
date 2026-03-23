"""GHOST-SIGINT FastAPI application."""
from __future__ import annotations
import logging, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config import get_settings
from api.routes.auth import router as auth_router
from api.routes.signals import router as signals_router
from api.routes.detections import router as detections_router
from api.routes.metrics import router as metrics_router
from api.websocket import router as ws_router
from constants import RATE_LIMIT_PER_MINUTE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")
logger = logging.getLogger("ghost-sigint")
settings = get_settings()
_rate_buckets: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("GHOST-SIGINT v1.0.0 starting")
    yield
    logger.info("GHOST-SIGINT shutting down")


app = FastAPI(
    title="GHOST-SIGINT",
    description="Military-grade RF Signal Intelligence System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    key = (request.cookies.get("access_token", "") or
           request.headers.get("authorization", "") or
           (request.client.host if request.client else ""))[:16]
    now = time.time()
    bucket = [t for t in _rate_buckets.get(key, []) if now - t < 60]
    if len(bucket) >= RATE_LIMIT_PER_MINUTE:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    bucket.append(now)
    _rate_buckets[key] = bucket
    return await call_next(request)


app.include_router(auth_router,       prefix="/api")
app.include_router(signals_router,    prefix="/api")
app.include_router(detections_router, prefix="/api")
app.include_router(metrics_router,    prefix="/api")
app.include_router(ws_router)

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        idx = os.path.join(frontend_dir, "index.html")
        return FileResponse(idx) if os.path.exists(idx) else {"message": "GHOST-SIGINT API"}


@app.get("/health")
async def health():
    return {"status": "operational", "system": "GHOST-SIGINT", "version": "1.0.0"}
