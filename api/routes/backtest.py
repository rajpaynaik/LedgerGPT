from __future__ import annotations

"""
API routes for backtesting and model training.
Provides endpoints for strategy validation and ML model management.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
import structlog

from backtest.backtester_enhanced import Backtester
from models.xgboost_signal_model import XGBoostSignalModel

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/backtest", tags=["backtest"])


@router.post("/run")
async def run_backtest(ticker: str = "AAPL", period: str = "1y") -> dict:
    """
    Run backtest on historical data for a ticker.
    
    Args:
        ticker: Stock ticker
        period: Historical period ('1y', '2y', '5y')
        
    Returns:
        Backtest results and performance metrics
    """
    try:
        backtester = Backtester()
        results = backtester.backtest(ticker, period)
        
        return {
            "ticker": results.ticker,
            "status": "completed",
            "metrics": results.to_dict(),
            "trades": backtester.get_trades_summary()
        }
    
    except Exception as e:
        logger.error("Backtest failed", ticker=ticker, error=str(e))
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@router.get("/metrics/{ticker}")
async def get_backtest_metrics(ticker: str, period: str = "1y") -> dict:
    """
    Get backtest metrics for a ticker without detailed trade list.
    
    Args:
        ticker: Stock ticker
        period: Historical period
        
    Returns:
        Performance metrics summary
    """
    try:
        backtester = Backtester()
        results = backtester.backtest(ticker, period)
        
        return {
            "ticker": results.ticker,
            "metrics": results.to_dict()
        }
    
    except Exception as e:
        logger.error("Metrics retrieval failed", ticker=ticker, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")
