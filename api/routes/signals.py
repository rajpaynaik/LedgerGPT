from __future__ import annotations

"""Trading signals endpoints."""
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, HTTPException, BackgroundTasks

from api.schemas import (
    BacktestRequest,
    BacktestResponse,
    SignalListResponse,
    SignalResponse,
)

router = APIRouter(prefix="/signals", tags=["signals"])
logger = structlog.get_logger(__name__)

# Shared predictor instance (model loaded once)
_predictor = None
_sentiment_worker = None


def set_sentiment_worker(worker) -> None:
    """Set the sentiment worker for signal generation."""
    global _sentiment_worker
    _sentiment_worker = worker


def _get_predictor():
    global _predictor
    if _predictor is None:
        from models.predict import SignalPredictor
        _predictor = SignalPredictor(sentiment_worker=_sentiment_worker)
    return _predictor


@router.get("/", response_model=SignalListResponse)
async def get_all_signals() -> SignalListResponse:
    """
    Generate live BUY/HOLD/SELL signals for all tracked tickers.
    Model loads on first call (~30s for FinLLaMA + XGBoost).
    """
    try:
        predictor = _get_predictor()
        signals = predictor.predict_all()
        return SignalListResponse(
            signals=[SignalResponse(**s) for s in signals],
            count=len(signals),
        )
    except Exception as exc:
        logger.error("signals_endpoint_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{ticker}", response_model=SignalResponse)
async def get_signal(ticker: str) -> SignalResponse:
    """Get signal for a specific ticker."""
    ticker = ticker.upper()
    try:
        predictor = _get_predictor()
        signal = predictor.predict_ticker(ticker)
        return SignalResponse(**signal)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """
    Run a historical backtest for the given tickers and period.
    Returns equity curve metrics and trade statistics.
    Note: this is a long-running operation (10-120s depending on period).
    """
    try:
        backtester = Backtester(
            initial_cash=request.initial_cash,
            commission=request.commission,
        )
        result = backtester.run(
            tickers=request.tickers,
            period=request.period,
            strategy_params={"min_confidence": request.min_confidence},
        )
        return BacktestResponse(**result)
    except Exception as exc:
        logger.error("backtest_endpoint_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/watchlist/tickers")
async def get_watchlist() -> dict:
    """Return the current tracked ticker watchlist."""
    return {"tickers": WATCHLIST, "count": len(WATCHLIST)}
