from __future__ import annotations

"""
Pydantic v2 request / response schemas for the REST API.
"""
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── Sentiment ────────────────────────────────────────────────────────────────
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=2048)
    extract_tickers: bool = True
    classify_event: bool = True


class SentimentResponse(BaseModel):
    ticker: Optional[str]
    all_tickers: list[str]  # type: ignore
    sentiment: Literal["bullish", "bearish", "neutral"]
    confidence: float
    impact: str
    reason: str
    event_type: Optional[str] = None
    urgency: Optional[str] = None
    affected_sectors: list[str] = []


# ── Signals ──────────────────────────────────────────────────────────────────
class SignalResponse(BaseModel):
    ticker: str
    signal: Literal["BUY", "HOLD", "SELL"]
    confidence: float
    probabilities: dict[str, float] = {}
    reason: str
    top_factors: list[dict] = []
    generated_at: datetime


class SignalListResponse(BaseModel):
    signals: list[SignalResponse]
    count: int


# ── Backtest ─────────────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    tickers: list[str] = Field(default=["TSLA", "AAPL"])
    period: str = Field(default="1y", pattern=r"^\d+[dwmy]$")
    initial_cash: float = Field(default=100_000, ge=1000)
    min_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    commission: float = Field(default=0.001, ge=0.0)


class EquityMetrics(BaseModel):
    total_return_pct: float
    annualised_return_pct: float
    annualised_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    var_95_pct: float
    cvar_95_pct: float
    start: str
    end: str


class TradeMetrics(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_pnl: float
    avg_pnl: float


class BacktestResponse(BaseModel):
    initial_cash: float
    final_value: float
    equity_metrics: EquityMetrics
    trade_metrics: TradeMetrics
    tickers: list[str]
    period: str


# ── Health ───────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    version: str
    checks: dict[str, str]
    timestamp: datetime
