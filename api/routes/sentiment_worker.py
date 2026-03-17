from __future__ import annotations

"""
Sentiment Worker API Endpoints
Exposes the sentiment generation pipeline via REST API.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field
import structlog

from llm.sentiment_worker import SentimentWorker
from database.sentiment_crud import SentimentDB

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/sentiment-worker", tags=["sentiment-worker"])

# Global worker instance (will be initialized by main.py)
sentiment_worker: Optional[SentimentWorker] = None
sentiment_db: Optional[SentimentDB] = None


def init_sentiment_worker(worker: SentimentWorker, db: SentimentDB) -> None:
    """Initialize sentiment worker (called from main.py)."""
    global sentiment_worker, sentiment_db
    sentiment_worker = worker
    sentiment_db = db


class TextAnalysisRequest(BaseModel):
    """Request to analyze financial text."""
    text: str = Field(..., min_length=10, max_length=5000, description="Financial text to analyze")
    source: str = Field(
        default="api",
        description="Text source: 'twitter', 'reddit', 'news', 'api'"
    )


class TextAnalysisResponse(BaseModel):
    """Response from text analysis."""
    tickers_detected: int
    results: dict
    analysis_timestamp: Optional[str] = None
    error: Optional[str] = None


class SentimentSummaryResponse(BaseModel):
    """Aggregated sentiment metrics for a ticker."""
    ticker: str
    mention_count: int
    avg_sentiment: float
    bullish_count: int
    bearish_count: int
    neutral_count: int
    avg_confidence: float


@router.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze financial text for sentiment.
    
    - Detects stock tickers ($AAPL, TSLA, etc.)
    - Calls FinLLM sentiment analysis
    - Stores results in database
    
    Returns sentiment analysis for each detected ticker.
    """
    if not sentiment_worker:
        raise HTTPException(status_code=503, detail="Sentiment worker not initialized")
    
    try:
        result = sentiment_worker.analyze_text(
            text=request.text,
            source=request.source
        )
        return TextAnalysisResponse(**result)
    except Exception as e:
        logger.error("Text analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/sentiment/{ticker}")
async def get_ticker_sentiment(
    ticker: str = Path(..., min_length=1, max_length=10),
    hours: int = Query(24, ge=1, le=720)
):
    """
    Get latest sentiment for a ticker.
    
    Args:
        ticker: Stock ticker (e.g., AAPL)
        hours: Lookback window in hours (default 24)
    """
    if not sentiment_worker:
        raise HTTPException(status_code=503, detail="Sentiment worker not initialized")
    
    sentiment = sentiment_worker.get_latest_sentiment(ticker)
    
    if not sentiment:
        raise HTTPException(
            status_code=404,
            detail=f"No sentiment data found for {ticker} in past {hours} hours"
        )
    
    return sentiment


@router.get("/summary/{ticker}", response_model=SentimentSummaryResponse)
async def get_sentiment_summary(
    ticker: str = Path(..., min_length=1, max_length=10),
    hours: int = Query(24, ge=1, le=720)
):
    """
    Get aggregated sentiment metrics for a ticker.
    
    Shows:
    - Number of mentions
    - Average sentiment score
    - Bullish/bearish/neutral breakdown
    - Average confidence
    """
    if not sentiment_worker:
        raise HTTPException(status_code=503, detail="Sentiment worker not initialized")
    
    summary = sentiment_worker.get_sentiment_summary(ticker)
    
    if not summary:
        raise HTTPException(
            status_code=404,
            detail=f"No sentiment data found for {ticker}"
        )
    
    return SentimentSummaryResponse(**summary)


@router.post("/batch-analyze")
async def analyze_batch(
    texts: list[str] = None,
    source: str = Query("api", description="Text source")
):
    """
    Analyze multiple texts in batch.
    
    Args:
        texts: List of financial texts (up to 50)
        source: Source identifier
    """
    if not sentiment_worker:
        raise HTTPException(status_code=503, detail="Sentiment worker not initialized")
    
    if not texts or len(texts) == 0:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    try:
        result = sentiment_worker.analyze_batch(texts, source=source)
        return result
    except Exception as e:
        logger.error("Batch analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.get("/trending")
async def get_trending_sentiments(
    limit: int = Query(10, ge=1, le=100),
    hours: int = Query(6, ge=1, le=168)
):
    """
    Get tickers with highest sentiment activity (trending).
    
    Args:
        limit: Number of tickers to return
        hours: Lookback window
    """
    if not sentiment_db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    trending = sentiment_db.get_trending_sentiments(limit=limit, hours=hours)
    
    if not trending:
        raise HTTPException(
            status_code=404,
            detail="No trending sentiment data found"
        )
    
    return {"trending": trending}
