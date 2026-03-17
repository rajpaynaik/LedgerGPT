from __future__ import annotations

"""
Twitter/X ingestion using Tweepy streaming API.
Filters by financial keywords and pushes raw posts to Kafka.
"""
import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import structlog
import tweepy

from config import get_settings
from stream.kafka_producer import KafkaProducerClient

logger = structlog.get_logger(__name__)

FINANCIAL_KEYWORDS = [
    "stock", "shares", "earnings", "bullish", "bearish",
    "NYSE", "NASDAQ", "SEC", "IPO", "options", "calls", "puts",
    "$TSLA", "$AAPL", "$AMZN", "$MSFT", "$NVDA", "$META",
    "short squeeze", "buy the dip", "market crash", "bull run",
]

TRACKED_CASHTAGS = [
    "TSLA", "AAPL", "AMZN", "MSFT", "NVDA", "META",
    "GOOGL", "SPY", "QQQ", "AMD", "BABA", "COIN",
]


class FinancialStreamListener(tweepy.StreamingClient):
    """Streams tweets and publishes raw payloads to Kafka."""

    def __init__(self, bearer_token: str, producer: KafkaProducerClient, **kwargs):
        super().__init__(bearer_token, **kwargs)
        self.producer = producer

    def on_tweet(self, tweet: tweepy.Tweet) -> None:
        payload = {
            "source": "twitter",
            "id": str(tweet.id),
            "text": tweet.text,
            "author_id": str(tweet.author_id),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw": tweet.data,
        }
        self.producer.publish(
            topic=get_settings().kafka_topic_raw_social,
            key=str(tweet.id),
            value=payload,
        )
        logger.debug("tweet_ingested", tweet_id=tweet.id)

    def on_errors(self, errors: Any) -> None:
        logger.error("twitter_stream_error", errors=errors)


class TwitterIngester:
    """Manages Twitter stream lifecycle."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.producer = KafkaProducerClient()
        self._stream: FinancialStreamListener | None = None

    # ── REST historical pull ────────────────────────────────────────────────
    def fetch_recent(self, query: str, max_results: int = 100) -> list[dict]:
        """Pull recent tweets matching a query (for backfill / backtesting)."""
        client = tweepy.Client(
            bearer_token=self.settings.twitter_bearer_token.get_secret_value()
        )
        response = client.search_recent_tweets(
            query=query,
            max_results=min(max_results, 100),
            tweet_fields=["created_at", "author_id", "public_metrics", "lang"],
        )
        tweets = []
        if response.data:
            for tweet in response.data:
                tweets.append({
                    "source": "twitter",
                    "id": str(tweet.id),
                    "text": tweet.text,
                    "author_id": str(tweet.author_id),
                    "created_at": str(tweet.created_at),
                    "metrics": tweet.public_metrics,
                })
        logger.info("twitter_fetch_recent", count=len(tweets), query=query)
        return tweets

    # ── Live streaming ──────────────────────────────────────────────────────
    def start_stream(self) -> None:
        bearer = self.settings.twitter_bearer_token.get_secret_value()
        self._stream = FinancialStreamListener(
            bearer_token=bearer,
            producer=self.producer,
        )
        # Clear old rules and add fresh ones
        existing = self._stream.get_rules()
        if existing.data:
            ids = [rule.id for rule in existing.data]
            self._stream.delete_rules(ids)

        rules = [
            tweepy.StreamRule(f"${tag} lang:en -is:retweet")
            for tag in TRACKED_CASHTAGS
        ]
        self._stream.add_rules(rules)
        logger.info("twitter_stream_starting", tickers=TRACKED_CASHTAGS)
        self._stream.filter(
            tweet_fields=["created_at", "author_id", "public_metrics"]
        )

    def stop_stream(self) -> None:
        if self._stream:
            self._stream.disconnect()
            logger.info("twitter_stream_stopped")
