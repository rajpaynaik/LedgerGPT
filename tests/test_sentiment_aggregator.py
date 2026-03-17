"""
Unit tests for SentimentAggregator.
No external dependencies required.
"""
from datetime import datetime, timedelta, timezone

import pytest

from features.sentiment_aggregator import SentimentAggregator


def make_record(
    ticker: str = "TSLA",
    sentiment: str = "bullish",
    confidence: float = 0.8,
    source: str = "twitter",
    minutes_ago: int = 30,
    urgency: str = "medium",
) -> dict:
    ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return {
        "id": f"{ticker}_{minutes_ago}",
        "source": source,
        "ticker": ticker,
        "all_tickers": [ticker],
        "sentiment": sentiment,
        "confidence": confidence,
        "urgency": urgency,
        "processed_at": ts.isoformat(),
    }


class TestSentimentAggregator:
    def setup_method(self):
        self.agg = SentimentAggregator(window_hours=24)

    def test_empty_records_returns_zeros(self):
        result = self.agg.aggregate([], "TSLA")
        assert result["overall_sentiment"] == 0.0
        assert result["tweet_count"] == 0

    def test_bullish_score_positive(self):
        records = [make_record("TSLA", "bullish", 0.9, "twitter") for _ in range(5)]
        result = self.agg.aggregate(records, "TSLA")
        assert result["twitter_sentiment"] > 0
        assert result["overall_sentiment"] > 0
        assert result["bullish_ratio"] == 1.0

    def test_bearish_score_negative(self):
        records = [make_record("TSLA", "bearish", 0.9, "reddit") for _ in range(5)]
        result = self.agg.aggregate(records, "TSLA")
        assert result["reddit_sentiment"] < 0
        assert result["bearish_ratio"] == 1.0

    def test_source_bucketing(self):
        records = [
            make_record("AAPL", "bullish", 0.8, "twitter"),
            make_record("AAPL", "bearish", 0.7, "reddit"),
            make_record("AAPL", "bullish", 0.9, "newsapi"),
        ]
        result = self.agg.aggregate(records, "AAPL")
        assert result["tweet_count"] == 1
        assert result["reddit_count"] == 1
        assert result["news_count"] == 1

    def test_filters_outside_window(self):
        old_record = make_record("TSLA", "bullish", 0.9, "twitter", minutes_ago=25 * 60)
        recent_record = make_record("TSLA", "bullish", 0.9, "twitter", minutes_ago=30)
        result = self.agg.aggregate([old_record, recent_record], "TSLA")
        assert result["tweet_count"] == 1  # only the recent one

    def test_filters_wrong_ticker(self):
        records = [make_record("AAPL", "bullish", 0.9)]
        result = self.agg.aggregate(records, "TSLA")
        assert result["tweet_count"] == 0

    def test_sentiment_momentum_positive_when_recent_bullish(self):
        # Older records: bearish
        old = [make_record("NVDA", "bearish", 0.8, minutes_ago=20 * 60) for _ in range(3)]
        # Recent records: bullish
        recent = [make_record("NVDA", "bullish", 0.9, minutes_ago=60) for _ in range(5)]
        result = self.agg.aggregate(old + recent, "NVDA")
        assert result["sentiment_momentum"] > 0
