"""
SQLAlchemy ORM models.
TimescaleDB hypertables are created via the migration SQL for time-series tables.
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class SentimentRecord(Base):
    """
    Processed sentiment output from FinLLaMA.
    Partitioned by processed_at (TimescaleDB hypertable).
    """
    __tablename__ = "sentiment_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(String(128), nullable=False, index=True)
    source = Column(String(50), nullable=False)   # twitter | reddit | news
    ticker = Column(String(10), index=True)
    all_tickers = Column(JSON, default=list)
    sentiment = Column(String(10))                # bullish | bearish | neutral
    confidence = Column(Float)
    impact = Column(String(30))
    reason = Column(Text)
    event_type = Column(String(40))
    urgency = Column(String(10))
    affected_sectors = Column(JSON, default=list)
    raw_text = Column(Text)
    processed_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("source_id", "source", name="uq_sentiment_source_id"),
        Index("ix_sentiment_ticker_time", "ticker", "processed_at"),
    )


class TradingSignal(Base):
    """ML model output — BUY / HOLD / SELL per ticker."""
    __tablename__ = "trading_signals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(10), nullable=False, index=True)
    signal = Column(String(5), nullable=False)    # BUY | HOLD | SELL
    confidence = Column(Float)
    prob_buy = Column(Float)
    prob_hold = Column(Float)
    prob_sell = Column(Float)
    reason = Column(Text)
    top_factors = Column(JSON, default=list)
    generated_at = Column(DateTime(timezone=True), nullable=False, index=True)
    executed = Column(Boolean, default=False)

    __table_args__ = (
        Index("ix_signal_ticker_time", "ticker", "generated_at"),
    )


class Order(Base):
    """Broker order execution record."""
    __tablename__ = "orders"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    broker_order_id = Column(String(128), unique=True)
    ticker = Column(String(10), nullable=False, index=True)
    side = Column(String(5))                       # BUY | SELL
    qty = Column(Float)
    price_at_signal = Column(Float)
    fill_price = Column(Float)
    status = Column(String(20))
    signal_confidence = Column(Float)
    signal_reason = Column(Text)
    submitted_at = Column(DateTime(timezone=True), index=True)
    filled_at = Column(DateTime(timezone=True))
    pnl = Column(Float)


class PriceBar(Base):
    """OHLCV price data. TimescaleDB hypertable on timestamp."""
    __tablename__ = "price_bars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    vwap = Column(Float)
    timeframe = Column(String(5), default="1d")   # 1m | 5m | 1h | 1d

    __table_args__ = (
        UniqueConstraint("ticker", "timestamp", "timeframe", name="uq_price_bar"),
        Index("ix_price_ticker_time", "ticker", "timestamp"),
    )
