from __future__ import annotations

"""
CRUD operations — async SQLAlchemy 2.0 style.
"""
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config import get_settings
from .models import Base, SentimentRecord, TradingSignal, Order, PriceBar

logger = structlog.get_logger(__name__)


def get_engine():
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )


def get_session_factory(engine=None):
    engine = engine or get_engine()
    return async_sessionmaker(engine, expire_on_commit=False)


# ── Sentiment ──────────────────────────────────────────────────────────────
class SentimentCRUD:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(self, record: dict) -> SentimentRecord:
        stmt = (
            pg_insert(SentimentRecord)
            .values(**record)
            .on_conflict_do_nothing(constraint="uq_sentiment_source_id")
            .returning(SentimentRecord)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.scalar_one_or_none()

    async def get_recent(
        self,
        ticker: str,
        hours: int = 24,
    ) -> list[SentimentRecord]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        result = await self.session.execute(
            select(SentimentRecord)
            .where(SentimentRecord.ticker == ticker)
            .where(SentimentRecord.processed_at >= cutoff)
            .order_by(SentimentRecord.processed_at.desc())
        )
        return result.scalars().all()

    async def get_sentiment_summary(self, ticker: str, hours: int = 24) -> dict:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        result = await self.session.execute(
            select(
                SentimentRecord.sentiment,
                func.count(SentimentRecord.id).label("count"),
                func.avg(SentimentRecord.confidence).label("avg_confidence"),
            )
            .where(SentimentRecord.ticker == ticker)
            .where(SentimentRecord.processed_at >= cutoff)
            .group_by(SentimentRecord.sentiment)
        )
        rows = result.all()
        return {r.sentiment: {"count": r.count, "avg_confidence": r.avg_confidence} for r in rows}


# ── Signals ────────────────────────────────────────────────────────────────
class SignalCRUD:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def insert(self, signal: dict) -> TradingSignal:
        obj = TradingSignal(**signal)
        self.session.add(obj)
        await self.session.commit()
        await self.session.refresh(obj)
        return obj

    async def get_latest(self, ticker: str) -> TradingSignal | None:
        result = await self.session.execute(
            select(TradingSignal)
            .where(TradingSignal.ticker == ticker)
            .order_by(TradingSignal.generated_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_history(self, ticker: str, limit: int = 100) -> list[TradingSignal]:
        result = await self.session.execute(
            select(TradingSignal)
            .where(TradingSignal.ticker == ticker)
            .order_by(TradingSignal.generated_at.desc())
            .limit(limit)
        )
        return result.scalars().all()


# ── Orders ─────────────────────────────────────────────────────────────────
class OrderCRUD:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def insert(self, order: dict) -> Order:
        obj = Order(**order)
        self.session.add(obj)
        await self.session.commit()
        await self.session.refresh(obj)
        return obj

    async def update_fill(
        self, broker_order_id: str, fill_price: float, status: str
    ) -> None:
        result = await self.session.execute(
            select(Order).where(Order.broker_order_id == broker_order_id)
        )
        obj = result.scalar_one_or_none()
        if obj:
            obj.fill_price = fill_price
            obj.status = status
            obj.filled_at = datetime.now(timezone.utc)
            await self.session.commit()

    async def get_open_positions(self) -> list[Order]:
        result = await self.session.execute(
            select(Order)
            .where(Order.status == "filled")
            .where(Order.side == "BUY")
        )
        return result.scalars().all()
