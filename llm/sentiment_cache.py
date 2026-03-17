from __future__ import annotations

"""
Redis-based sentiment caching layer.
Caches recent sentiment analysis to reduce database queries.
"""

import json
from datetime import datetime, timedelta
from typing import Optional
import redis
import structlog

logger = structlog.get_logger(__name__)


class SentimentCache:
    """
    Redis-based cache for sentiment analysis results.
    TTL: 10 minutes for recent sentiment, 1 hour for aggregates.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize Redis sentiment cache.
        
        Args:
            redis_url: Redis connection URL
        """
        try:
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self.redis.ping()
            self.enabled = True
            logger.info("Sentiment cache initialized", redis_url=redis_url)
        except Exception as e:
            self.redis = None
            self.enabled = False
            logger.warning("Sentiment cache unavailable", error=str(e))
    
    def _get_key(self, key_type: str, ticker: str, params: str = "") -> str:
        """Build cache key."""
        if params:
            return f"sentiment:{key_type}:{ticker}:{params}"
        return f"sentiment:{key_type}:{ticker}"
    
    def cache_latest_sentiment(
        self,
        ticker: str,
        sentiment_data: dict,
        ttl: int = 600  # 10 minutes
    ) -> bool:
        """
        Cache latest sentiment analysis result.
        
        Args:
            ticker: Stock ticker
            sentiment_data: Sentiment analysis dict
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        if not self.enabled:
            return False
        
        try:
            key = self._get_key("latest", ticker)
            value = json.dumps(sentiment_data)
            self.redis.setex(key, ttl, value)
            logger.debug("Cached latest sentiment", ticker=ticker, ttl=ttl)
            return True
        except Exception as e:
            logger.error("Cache set failed", error=str(e), ticker=ticker)
            return False
    
    def get_latest_sentiment(self, ticker: str) -> Optional[dict]:
        """
        Retrieve cached latest sentiment.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Cached sentiment dict or None
        """
        if not self.enabled:
            return None
        
        try:
            key = self._get_key("latest", ticker)
            value = self.redis.get(key)
            if value:
                data = json.loads(value)
                logger.debug("Hit: latest sentiment cache", ticker=ticker)
                return data
            return None
        except Exception as e:
            logger.error("Cache get failed", error=str(e), ticker=ticker)
            return None
    
    def cache_sentiment_summary(
        self,
        ticker: str,
        summary: dict,
        ttl: int = 3600  # 1 hour
    ) -> bool:
        """
        Cache aggregated sentiment summary.
        
        Args:
            ticker: Stock ticker
            summary: Aggregated sentiment summary dict
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        if not self.enabled:
            return False
        
        try:
            key = self._get_key("summary", ticker)
            value = json.dumps(summary)
            self.redis.setex(key, ttl, value)
            logger.debug("Cached sentiment summary", ticker=ticker, ttl=ttl)
            return True
        except Exception as e:
            logger.error("Cache set failed", error=str(e), ticker=ticker)
            return False
    
    def get_sentiment_summary(self, ticker: str) -> Optional[dict]:
        """
        Retrieve cached sentiment summary.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Cached summary dict or None
        """
        if not self.enabled:
            return None
        
        try:
            key = self._get_key("summary", ticker)
            value = self.redis.get(key)
            if value:
                data = json.loads(value)
                logger.debug("Hit: sentiment summary cache", ticker=ticker)
                return data
            return None
        except Exception as e:
            logger.error("Cache get failed", error=str(e), ticker=ticker)
            return None
    
    def cache_trending_tickers(
        self,
        trending: list,
        ttl: int = 3600  # 1 hour
    ) -> bool:
        """
        Cache trending tickers list.
        
        Args:
            trending: List of trending ticker dicts
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        if not self.enabled:
            return False
        
        try:
            key = self._get_key("trending", "list")
            value = json.dumps(trending)
            self.redis.setex(key, ttl, value)
            logger.debug("Cached trending tickers", count=len(trending), ttl=ttl)
            return True
        except Exception as e:
            logger.error("Cache set failed", error=str(e))
            return False
    
    def get_trending_tickers(self) -> Optional[list]:
        """
        Retrieve cached trending tickers.
        
        Returns:
            Cached trending list or None
        """
        if not self.enabled:
            return None
        
        try:
            key = self._get_key("trending", "list")
            value = self.redis.get(key)
            if value:
                data = json.loads(value)
                logger.debug("Hit: trending tickers cache")
                return data
            return None
        except Exception as e:
            logger.error("Cache get failed", error=str(e))
            return None
    
    def invalidate_ticker(self, ticker: str) -> bool:
        """
        Invalidate all cache entries for a ticker.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Success status
        """
        if not self.enabled:
            return False
        
        try:
            pattern = f"sentiment:*:{ticker}*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
                logger.debug("Invalidated ticker cache", ticker=ticker, keys=len(keys))
            return True
        except Exception as e:
            logger.error("Cache invalidation failed", error=str(e), ticker=ticker)
            return False
    
    def invalidate_all(self) -> bool:
        """Invalidate all sentiment cache."""
        if not self.enabled:
            return False
        
        try:
            pattern = "sentiment:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
                logger.debug("Invalidated all sentiment cache", keys=len(keys))
            return True
        except Exception as e:
            logger.error("Cache invalidation failed", error=str(e))
            return False
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            keys = self.redis.keys("sentiment:*")
            info = self.redis.info("memory")
            return {
                "enabled": True,
                "total_keys": len(keys),
                "memory_usage": info.get("used_memory_human", "N/A"),
                "memory_percent": info.get("used_memory_percent", "N/A")
            }
        except Exception as e:
            logger.error("Stats retrieval failed", error=str(e))
            return {"enabled": True, "error": str(e)}
