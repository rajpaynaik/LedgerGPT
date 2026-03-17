from __future__ import annotations

"""
FastAPI application factory.
Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from config import get_settings
from api.routes import health_router, sentiment_router, signals_router
from llm.sentiment_worker import SentimentWorker
from llm.sentiment_cache import SentimentCache
from database.sentiment_crud import SentimentDB

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ──────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "ledgergpt_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "ledgergpt_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "ledgergpt_starting",
        env=settings.app_env,
        log_level=settings.log_level,
    )
    
    # Initialize sentiment worker and database
    try:
        sentiment_db = SentimentDB()
        sentiment_cache = SentimentCache()
        sentiment_worker = SentimentWorker(db=sentiment_db, cache=sentiment_cache)
        
        # Register with router
        from api.routes.sentiment_worker import init_sentiment_worker
        init_sentiment_worker(sentiment_worker, sentiment_db)
        
        # Register with signal predictor
        from api.routes.signals import set_sentiment_worker
        set_sentiment_worker(sentiment_worker)
        
        logger.info("Sentiment worker initialized with caching")
    except Exception as e:
        logger.warning("Failed to initialize sentiment worker", error=str(e))
    
    yield
    logger.info("ledgergpt_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="LedgerGPT Trading Intelligence API",
        description=(
            "AI-powered trading signal system using FinLLaMA for sentiment analysis "
            "and XGBoost for signal generation."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ───────────────────────────────────────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        response: Response = await call_next(request)
        latency = time.perf_counter() - start
        endpoint = request.url.path
        REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
        REQUEST_LATENCY.labels(endpoint).observe(latency)
        return response

    # ── Routers ──────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(sentiment_router, prefix="/api/v1")
    app.include_router(signals_router, prefix="/api/v1")
    
    # Import and include sentiment worker router
    from api.routes.sentiment_worker import router as sentiment_worker_router
    app.include_router(sentiment_worker_router)
    
    # Import and include backtest router
    from api.routes.backtest import router as backtest_router
    app.include_router(backtest_router)
    
    # Import and include ML model router
    from api.routes.ml_model import router as ml_model_router
    app.include_router(ml_model_router)

    # ── Prometheus scrape endpoint ───────────────────────────────────────────
    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics():
        from fastapi.responses import Response as FastResponse
        return FastResponse(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


app = create_app()
