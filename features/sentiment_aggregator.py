from __future__ import annotations

"""
Aggregates raw sentiment records (from Kafka / DB) into time-bucketed
sentiment feature vectors per ticker.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

SENTIMENT_MAP = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
IMPACT_MAP = {
    "short_term_positive": 0.8,
    "long_term_positive": 0.5,
    "neutral": 0.0,
    "short_term_negative": -0.8,
    "long_term_negative": -0.5,
}
URGENCY_MAP = {"high": 1.0, "medium": 0.5, "low": 0.2}


class SentimentAggregator:
    """
    Takes a list of SentimentRecord dicts (from Kafka topic processed.sentiment)
    and builds aggregated features per ticker per time window.
    """

    def __init__(self, window_hours: int = 24) -> None:
        self.window_hours = window_hours

    def _score(self, record: dict) -> float:
        """Convert a sentiment record to a single scalar score [-1, 1]."""
        s = SENTIMENT_MAP.get(record.get("sentiment", "neutral"), 0.0)
        c = float(record.get("confidence", 0.5))
        u = URGENCY_MAP.get(record.get("urgency", "low"), 0.2)
        return s * c * u

    def aggregate(
        self,
        records: list[dict],
        ticker: str,
        as_of: datetime | None = None,
    ) -> dict:
        """
        Produce a feature dict for a given ticker from a list of sentiment records.

        Returns:
            {
                "ticker": "TSLA",
                "twitter_sentiment": float,
                "reddit_sentiment": float,
                "news_sentiment": float,
                "overall_sentiment": float,
                "tweet_count": int,
                "reddit_count": int,
                "news_count": int,
                "avg_confidence": float,
                "high_urgency_count": int,
                "bullish_ratio": float,
                "bearish_ratio": float,
                "sentiment_momentum": float,  # recent vs older window
                "as_of": str,
            }
        """
        as_of = as_of or datetime.now(timezone.utc)
        cutoff = as_of - timedelta(hours=self.window_hours)
        half = as_of - timedelta(hours=self.window_hours // 2)

        # Filter to ticker within window
        filtered = []
        for r in records:
            tickers = r.get("all_tickers", [])
            if r.get("ticker") == ticker or ticker in tickers:
                try:
                    ts = datetime.fromisoformat(r["processed_at"].replace("Z", "+00:00"))
                    if ts >= cutoff:
                        r["_ts"] = ts
                        r["_score"] = self._score(r)
                        filtered.append(r)
                except (KeyError, ValueError):
                    pass

        if not filtered:
            return self._empty(ticker, as_of)

        # Source buckets
        by_source: dict[str, list[dict]] = {"twitter": [], "reddit": [], "news": []}
        for r in filtered:
            src = r.get("source", "")
            if "twitter" in src:
                by_source["twitter"].append(r)
            elif "reddit" in src:
                by_source["reddit"].append(r)
            else:
                by_source["news"].append(r)

        def avg_score(recs: list[dict]) -> float:
            if not recs:
                return 0.0
            return float(np.mean([r["_score"] for r in recs]))

        def bullish_ratio(recs: list[dict]) -> float:
            if not recs:
                return 0.0
            return sum(1 for r in recs if r.get("sentiment") == "bullish") / len(recs)

        def bearish_ratio(recs: list[dict]) -> float:
            if not recs:
                return 0.0
            return sum(1 for r in recs if r.get("sentiment") == "bearish") / len(recs)

        all_scores = [r["_score"] for r in filtered]
        recent = [r["_score"] for r in filtered if r["_ts"] >= half]
        older = [r["_score"] for r in filtered if r["_ts"] < half]
        momentum = (np.mean(recent) if recent else 0.0) - (np.mean(older) if older else 0.0)

        return {
            "ticker": ticker,
            "twitter_sentiment": avg_score(by_source["twitter"]),
            "reddit_sentiment": avg_score(by_source["reddit"]),
            "news_sentiment": avg_score(by_source["news"]),
            "overall_sentiment": float(np.mean(all_scores)),
            "tweet_count": len(by_source["twitter"]),
            "reddit_count": len(by_source["reddit"]),
            "news_count": len(by_source["news"]),
            "avg_confidence": float(np.mean([r.get("confidence", 0.5) for r in filtered])),
            "high_urgency_count": sum(1 for r in filtered if r.get("urgency") == "high"),
            "bullish_ratio": bullish_ratio(filtered),
            "bearish_ratio": bearish_ratio(filtered),
            "sentiment_momentum": float(momentum),
            "as_of": as_of.isoformat(),
        }

    def aggregate_dataframe(
        self,
        records: list[dict],
        tickers: list[str],
        freq: str = "1H",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Build a time-series DataFrame of sentiment features for backtesting.
        One row per (ticker, timestamp) bucket.
        """
        rows = []
        end = end or datetime.now(timezone.utc)
        start = start or (end - timedelta(days=30))
        current = start

        while current <= end:
            for ticker in tickers:
                agg = self.aggregate(records, ticker, as_of=current)
                agg["timestamp"] = current
                rows.append(agg)
            current += pd.Timedelta(freq)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index(["timestamp", "ticker"])
        return df

    def _empty(self, ticker: str, as_of: datetime) -> dict:
        return {
            "ticker": ticker,
            "twitter_sentiment": 0.0,
            "reddit_sentiment": 0.0,
            "news_sentiment": 0.0,
            "overall_sentiment": 0.0,
            "tweet_count": 0,
            "reddit_count": 0,
            "news_count": 0,
            "avg_confidence": 0.0,
            "high_urgency_count": 0,
            "bullish_ratio": 0.0,
            "bearish_ratio": 0.0,
            "sentiment_momentum": 0.0,
            "as_of": as_of.isoformat(),
        }
