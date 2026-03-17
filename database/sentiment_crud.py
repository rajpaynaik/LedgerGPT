from __future__ import annotations

"""
Database CRUD operations for sentiment analysis storage.
Handles reading/writing sentiment results from FinLLM.
"""

from datetime import datetime, timedelta
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)


class SentimentDB:
    """
    Sentiment data persistence layer.
    Stores and retrieves FinLLM sentiment analysis results.
    """
    
    def __init__(self, db_connection = None):
        """Initialize with database connection (will be set by app)."""
        self.conn = db_connection
    
    def store_sentiment(
        self,
        ticker: str,
        sentiment_score: float,
        confidence: float,
        source: str,
        text_content: str = None,
        event_type: str = None,
        event_impact: str = None,
    ) -> bool:
        """
        Store sentiment analysis result in database.
        
        Args:
            ticker: Stock ticker (e.g., 'AAPL')
            sentiment_score: -1.0 (bearish) to +1.0 (bullish)
            confidence: 0.0 to 1.0 model confidence
            source: 'twitter', 'reddit', 'news', 'api'
            text_content: Original text analyzed
            event_type: 'earnings', 'product', 'macro', 'rumor', 'general'
            event_impact: 'short_term', 'long_term'
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.conn:
            logger.warning("Database not connected, sentiment not stored", ticker=ticker)
            return False
        
        try:
            query = """
            INSERT INTO sentiment_analysis 
            (ticker, sentiment_score, confidence, source, text_content, event_type, event_impact)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            self.conn.execute(query, (
                ticker.upper(),
                float(sentiment_score),
                float(confidence),
                source,
                text_content,
                event_type,
                event_impact
            ))
            self.conn.commit()
            logger.info("Sentiment stored", ticker=ticker, score=sentiment_score, confidence=confidence)
            return True
        except Exception as e:
            logger.error("Failed to store sentiment", ticker=ticker, error=str(e))
            return False
    
    def get_latest_sentiment(self, ticker: str, hours: int = 24) -> Optional[dict]:
        """
        Get the most recent sentiment for a ticker.
        
        Args:
            ticker: Stock ticker
            hours: Look back window (default 24 hours)
            
        Returns:
            Latest sentiment record or None if not found
        """
        if not self.conn:
            return None
        
        try:
            query = """
            SELECT ticker, sentiment_score, confidence, source, event_type, 
                   event_impact, created_at
            FROM sentiment_analysis
            WHERE ticker = %s AND created_at > NOW() - INTERVAL '%s hours'
            ORDER BY created_at DESC
            LIMIT 1
            """
            cursor = self.conn.cursor()
            cursor.execute(query, (ticker.upper(), hours))
            row = cursor.fetchone()
            
            if row:
                return {
                    'ticker': row[0],
                    'sentiment_score': float(row[1]),
                    'confidence': float(row[2]),
                    'source': row[3],
                    'event_type': row[4],
                    'event_impact': row[5],
                    'created_at': row[6]
                }
            return None
        except Exception as e:
            logger.error("Failed to fetch sentiment", ticker=ticker, error=str(e))
            return None
    
    def get_sentiment_aggregate(self, ticker: str, hours: int = 24) -> Optional[dict]:
        """
        Get aggregated sentiment metrics for a ticker.
        
        Args:
            ticker: Stock ticker
            hours: Lookback window
            
        Returns:
            Aggregated sentiment metrics
        """
        if not self.conn:
            return None
        
        try:
            query = """
            SELECT 
                COUNT(*) as mention_count,
                AVG(sentiment_score) as avg_sentiment,
                MAX(sentiment_score) as max_sentiment,
                MIN(sentiment_score) as min_sentiment,
                AVG(confidence) as avg_confidence,
                COUNT(CASE WHEN sentiment_score > 0.3 THEN 1 END) as bullish_count,
                COUNT(CASE WHEN sentiment_score < -0.3 THEN 1 END) as bearish_count,
                COUNT(CASE WHEN sentiment_score BETWEEN -0.3 AND 0.3 THEN 1 END) as neutral_count
            FROM sentiment_analysis
            WHERE ticker = %s AND created_at > NOW() - INTERVAL '%s hours'
            """
            cursor = self.conn.cursor()
            cursor.execute(query, (ticker.upper(), hours))
            row = cursor.fetchone()
            
            if row and row[0] > 0:
                return {
                    'ticker': ticker.upper(),
                    'mention_count': int(row[0]),
                    'avg_sentiment': float(row[1]) if row[1] else 0.0,
                    'max_sentiment': float(row[2]) if row[2] else 0.0,
                    'min_sentiment': float(row[3]) if row[3] else 0.0,
                    'avg_confidence': float(row[4]) if row[4] else 0.0,
                    'bullish_count': int(row[5]),
                    'bearish_count': int(row[6]),
                    'neutral_count': int(row[7]),
                    'sentiment_velocity': 0.0  # Would calculate based on trends
                }
            return None
        except Exception as e:
            logger.error("Failed to fetch aggregated sentiment", ticker=ticker, error=str(e))
            return None
    
    def get_sentiment_by_source(self, ticker: str, hours: int = 24) -> dict[str, list]:
        """
        Get sentiment broken down by source (twitter, reddit, news).
        
        Args:
            ticker: Stock ticker
            hours: Lookback window
            
        Returns:
            Dict mapping source to sentiment records
        """
        if not self.conn:
            return {}
        
        try:
            query = """
            SELECT source, sentiment_score, confidence, event_type, created_at
            FROM sentiment_analysis
            WHERE ticker = %s AND created_at > NOW() - INTERVAL '%s hours'
            ORDER BY source, created_at DESC
            """
            cursor = self.conn.cursor()
            cursor.execute(query, (ticker.upper(), hours))
            rows = cursor.fetchall()
            
            results = {}
            for row in rows:
                source = row[0]
                if source not in results:
                    results[source] = []
                results[source].append({
                    'sentiment_score': float(row[1]),
                    'confidence': float(row[2]),
                    'event_type': row[3],
                    'created_at': row[4]
                })
            return results
        except Exception as e:
            logger.error("Failed to fetch sentiment by source", ticker=ticker, error=str(e))
            return {}
    
    def get_trending_sentiments(self, limit: int = 10, hours: int = 6) -> list[dict]:
        """
        Get tickers with highest sentiment velocity (trending).
        
        Args:
            limit: Number of tickers to return
            hours: Lookback window
            
        Returns:
            List of trending tickers with sentiment metrics
        """
        if not self.conn:
            return []
        
        try:
            query = """
            SELECT 
                ticker,
                COUNT(*) as mention_count,
                AVG(sentiment_score) as avg_sentiment,
                AVG(confidence) as avg_confidence
            FROM sentiment_analysis
            WHERE created_at > NOW() - INTERVAL '%s hours'
            GROUP BY ticker
            ORDER BY mention_count DESC
            LIMIT %s
            """
            cursor = self.conn.cursor()
            cursor.execute(query, (hours, limit))
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    'ticker': row[0],
                    'mention_count': int(row[1]),
                    'avg_sentiment': float(row[2]) if row[2] else 0.0,
                    'avg_confidence': float(row[3]) if row[3] else 0.0
                })
            return results
        except Exception as e:
            logger.error("Failed to fetch trending sentiments", error=str(e))
            return []
