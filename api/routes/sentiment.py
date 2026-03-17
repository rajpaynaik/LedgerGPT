from __future__ import annotations

"""Sentiment analysis endpoints."""
import structlog
from fastapi import APIRouter, HTTPException, BackgroundTasks

from api.schemas import SentimentRequest, SentimentResponse

router = APIRouter(prefix="/sentiment", tags=["sentiment"])
logger = structlog.get_logger(__name__)


def _get_llm():
    try:
        from llm.finllama_service import FinLLaMAService
        return FinLLaMAService.get_instance()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"LLM service unavailable: {exc}")


@router.post("/analyse", response_model=SentimentResponse)
async def analyse_sentiment(request: SentimentRequest) -> SentimentResponse:
    """
    Analyse the sentiment of a piece of financial text using FinLLaMA.
    Returns structured sentiment with ticker, confidence, and reasoning.
    """
    llm = _get_llm()

    sentiment = llm.analyse_sentiment(request.text)
    tickers = llm.extract_tickers(request.text) if request.extract_tickers else []
    event = llm.classify_event(request.text) if request.classify_event else {}

    return SentimentResponse(
        ticker=sentiment.get("ticker"),
        all_tickers=tickers,
        sentiment=sentiment.get("sentiment", "neutral"),
        confidence=sentiment.get("confidence", 0.5),
        impact=sentiment.get("impact", "neutral"),
        reason=sentiment.get("reason", ""),
        event_type=event.get("event_type"),
        urgency=event.get("urgency"),
        affected_sectors=event.get("affected_sectors", []),
    )


@router.post("/batch")
async def analyse_batch(texts: list[str]) -> list[SentimentResponse]:
    """
    Batch sentiment analysis for up to 50 texts.
    More efficient than individual /analyse calls.
    """
    if len(texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 texts per batch.")
    if not texts:
        return []

    llm = _get_llm()
    results = llm.analyse_batch(texts)

    return [
        SentimentResponse(
            ticker=r.get("ticker"),
            all_tickers=[r["ticker"]] if r.get("ticker") else [],
            sentiment=r.get("sentiment", "neutral"),
            confidence=float(r.get("confidence", 0.5)),
            impact=r.get("impact", "neutral"),
            reason=r.get("reason", ""),
        )
        for r in results
    ]
