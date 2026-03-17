from __future__ import annotations

"""
Sentiment Worker Service
Processes financial text, detects tickers, and calls FinLLM for sentiment analysis.
Stores results in database for signal generation.
"""

from datetime import datetime
from typing import Optional
import structlog
import requests
import json

from config import get_settings
from features.ticker_detection import detect_tickers
from features.event_detector import EventDetector
from database.sentiment_crud import SentimentDB
from llm.sentiment_cache import SentimentCache

logger = structlog.get_logger(__name__)


class SentimentWorker:
    """
    Worker service that generates sentiment signals using FinLLM.
    
    Flow:
    1. Receive text (from Twitter, Reddit, News API, or direct API call)
    2. Detect stock tickers in the text
    3. Call FinLLM /sentiment/analyse endpoint
    4. Store results in database
    5. Signal predictor retrieves latest sentiment for tickers
    """
    
    def __init__(self, db: SentimentDB = None, cache: SentimentCache = None):
        """
        Initialize sentiment worker.
        
        Args:
            db: SentimentDB instance for storage (optional for testing)
            cache: SentimentCache instance for Redis caching (optional)
        """
        self.settings = get_settings()
        self.db = db
        self.cache = cache or SentimentCache()
        self.event_detector = EventDetector()
        self.finllm_endpoint = "http://localhost:8000/api/v1/sentiment/analyse"
        self.batch_endpoint = "http://localhost:8000/api/v1/sentiment/batch"
        self.in_memory_cache = {}  # Fallback in-memory cache
    
    def analyze_text(
        self,
        text: str,
        source: str = "api",
        batch: bool = False
    ) -> dict:
        """
        Analyze financial text for sentiment and store results.
        
        Args:
            text: Financial text (tweet, news headline, post, etc.)
            source: Source identifier ('twitter', 'reddit', 'news', 'api')
            batch: Whether this is part of a batch request
            
        Returns:
            Dict with analysis results for each detected ticker
        """
        try:
            # Step 1: Detect tickers in the text
            tickers = detect_tickers(text)
            
            if not tickers:
                logger.debug("No tickers detected in text", source=source, text_len=len(text))
                return {"tickers_detected": 0, "results": {}}
            
            logger.info("Tickers detected", source=source, tickers=list(tickers), text_len=len(text))
            
            # Step 2: Call FinLLM sentiment endpoint
            sentiment_data = self._call_finllm(text)
            
            if not sentiment_data:
                logger.warning("FinLLM returned no sentiment data", source=source)
                return {
                    "tickers_detected": len(tickers),
                    "finllm_error": True,
                    "results": {}
                }
            
            # Step 2b: Detect event type
            event_detection = self.event_detector.detect_event_type(text)
            sentiment_data['event_type'] = event_detection['event_type']
            sentiment_data['event_impact'] = event_detection['impact']
            
            # Step 3: Store sentiment for each detected ticker
            results = {}
            for ticker in tickers:
                stored = self._store_sentiment_result(
                    ticker=ticker,
                    sentiment_data=sentiment_data,
                    source=source,
                    text_content=text[:1000] if source == "api" else text  # Truncate for DB
                )
                results[ticker] = stored
            
            # Step 4: Cache results for each ticker
            for ticker in tickers:
                if ticker in results:
                    latest_sentiment = self.db.get_latest_sentiment(ticker) if self.db else None
                    if latest_sentiment:
                        self.cache.cache_latest_sentiment(ticker, latest_sentiment)
            
            return {
                "tickers_detected": len(tickers),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "results": results
            }
        
        except Exception as e:
            logger.error("Failed to analyze text", source=source, error=str(e))
            return {"error": str(e), "results": {}}
    
    def _call_finllm(self, text: str) -> Optional[dict]:
        """
        Call FinLLM sentiment analysis endpoint.
        
        Args:
            text: Financial text to analyze
            
        Returns:
            Parsed JSON response from FinLLM or None
        """
        try:
            payload = {"text": text}
            response = requests.post(
                self.finllm_endpoint,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.debug("FinLLM sentiment received", status=response.status_code)
                return data
            else:
                logger.warning(
                    "FinLLM returned error",
                    status=response.status_code,
                    response=response.text[:200]
                )
                return None
        
        except requests.exceptions.Timeout:
            logger.error("FinLLM request timed out")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to FinLLM endpoint")
            return None
        except Exception as e:
            logger.error("Failed to call FinLLM", error=str(e))
            return None
    
    def _store_sentiment_result(
        self,
        ticker: str,
        sentiment_data: dict,
        source: str,
        text_content: str
    ) -> dict:
        """
        Store sentiment result in database.
        
        Args:
            ticker: Stock ticker
            sentiment_data: Response from FinLLM
            source: Data source
            text_content: Original text
            
        Returns:
            Stored record metadata
        """
        try:
            # Extract sentiment components from FinLLM response
            # Expected format: {sentiment_score, confidence, event_type, event_impact}
            sentiment_score = sentiment_data.get("sentiment_score", 0.0)
            confidence = sentiment_data.get("confidence", 0.5)
            event_type = sentiment_data.get("event_type", "general")
            event_impact = sentiment_data.get("event_impact", "short_term")
            
            # Store in database
            if self.db:
                success = self.db.store_sentiment(
                    ticker=ticker,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    source=source,
                    text_content=text_content,
                    event_type=event_type,
                    event_impact=event_impact
                )
                
                return {
                    "ticker": ticker,
                    "stored": success,
                    "sentiment_score": sentiment_score,
                    "confidence": confidence,
                    "event_type": event_type
                }
            else:
                return {
                    "ticker": ticker,
                    "stored": False,
                    "reason": "Database not initialized"
                }
        
        except Exception as e:
            logger.error("Failed to store sentiment", ticker=ticker, error=str(e))
            return {
                "ticker": ticker,
                "stored": False,
                "error": str(e)
            }
    
    def get_latest_sentiment(self, ticker: str) -> Optional[dict]:
        """
        Get latest sentiment for a ticker (cache-optimized).
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Latest sentiment record or None
        """
        # Try cache first
        cached = self.cache.get_latest_sentiment(ticker)
        if cached:
            return cached
        
        # Fall back to database
        if not self.db:
            return None
        
        result = self.db.get_latest_sentiment(ticker, hours=24)
        
        # Cache the result
        if result:
            self.cache.cache_latest_sentiment(ticker, result)
        
        return result
    
    def get_sentiment_summary(self, ticker: str) -> Optional[dict]:
        """
        Get aggregated sentiment metrics for a ticker (cache-optimized).
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Aggregated sentiment statistics
        """
        # Try cache first
        cached = self.cache.get_sentiment_summary(ticker)
        if cached:
            return cached
        
        # Fall back to database
        if not self.db:
            return None
        
        result = self.db.get_sentiment_aggregate(ticker, hours=24)
        
        # Cache the result
        if result:
            self.cache.cache_sentiment_summary(ticker, result)
        
        return result
    
    def analyze_batch(self, texts: list[str], source: str = "api") -> dict:
        """
        Analyze multiple texts at once.
        
        Args:
            texts: List of financial texts
            source: Source identifier
            
        Returns:
            Aggregated results
        """
        results = {
            "total_texts": len(texts),
            "analyses": [],
            "tickers_total": set()
        }
        
        for i, text in enumerate(texts[:50]):  # Limit to 50 for performance
            analysis = self.analyze_text(text, source=source, batch=True)
            results["analyses"].append(analysis)
            
            # Aggregate detected tickers
            for result in analysis.get("results", {}).values():
                if result.get("stored"):
                    results["tickers_total"].add(result.get("ticker"))
        
        results["tickers_total"] = list(results["tickers_total"])
        return results
