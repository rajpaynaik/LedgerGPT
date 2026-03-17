from __future__ import annotations

"""Health check endpoint."""
from datetime import datetime, timezone

import redis
import structlog
from fastapi import APIRouter

from api.schemas import HealthResponse
from config import get_settings

router = APIRouter(tags=["health"])
logger = structlog.get_logger(__name__)

APP_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    checks: dict[str, str] = {}

    # Redis (optional - not required for Swagger/API to work)
    try:
        r = redis.from_url(settings.redis_url, socket_connect_timeout=2)
        r.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = "unavailable (optional service)"
        logger.info("redis_unavailable", reason=str(exc))

    # DB (optional - not required for basic API)
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine(settings.database_url)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
        await engine.dispose()
    except Exception as exc:
        checks["database"] = "unavailable (optional service)"
        logger.info("database_unavailable", reason=str(exc))

    # API is considered healthy if the service is running (no hard dependencies)
    return HealthResponse(
        status="healthy",
        version=APP_VERSION,
        checks=checks,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/ping")
async def ping() -> dict:
    return {"pong": True}
