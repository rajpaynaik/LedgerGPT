from __future__ import annotations

"""
Sentiment Engine — orchestrates FinLLaMA inference over a Kafka stream.
Consumes raw.social and raw.news, produces processed.sentiment.
"""
import asyncio
from datetime import datetime, timezone
from typing import Any

import structlog

from config import get_settings
from stream.kafka_consumer import KafkaConsumerClient
from stream.kafka_producer import KafkaProducerClient
from stream.deduplication import DeduplicationFilter
from .finllama_service import FinLLaMAService

logger = structlog.get_logger(__name__)

# Minimum text length to bother sending to LLM
MIN_TEXT_LEN = 20


class SentimentEngine:
    """
    Consumes raw messages → calls FinLLaMA → publishes SentimentRecord to Kafka.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.producer = KafkaProducerClient()
        self.dedup = DeduplicationFilter()
        self._llm: FinLLaMAService | None = None

    def _get_llm(self) -> FinLLaMAService:
        if self._llm is None:
            self._llm = FinLLaMAService.get_instance()
        return self._llm

    def _extract_text(self, payload: dict) -> str:
        """Pull meaningful text from a raw social or news payload."""
        parts = []
        for field in ("title", "text", "selftext"):
            val = payload.get(field, "")
            if val and val.strip():
                parts.append(val.strip())
        return " ".join(parts)[:1024]  # cap at 1 KB

    def process_message(self, payload: dict, key: str) -> None:
        """Handler called by KafkaConsumerClient for each message."""
        if not self.dedup.is_new(payload):
            return

        text = self._extract_text(payload)
        if len(text) < MIN_TEXT_LEN:
            return

        try:
            llm = self._get_llm()
            sentiment = llm.analyse_sentiment(text)
            tickers = llm.extract_tickers(text)
            event = llm.classify_event(text)

            record = {
                "id": payload.get("id", ""),
                "source": payload.get("source", "unknown"),
                "ticker": sentiment.get("ticker") or (tickers[0] if tickers else None),
                "all_tickers": tickers,
                "sentiment": sentiment.get("sentiment", "neutral"),
                "confidence": sentiment.get("confidence", 0.5),
                "impact": sentiment.get("impact", "neutral"),
                "reason": sentiment.get("reason", ""),
                "event_type": event.get("event_type", "general_news"),
                "urgency": event.get("urgency", "low"),
                "affected_sectors": event.get("affected_sectors", []),
                "raw_text": text[:200],
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }

            self.producer.publish(
                topic=self.settings.kafka_topic_sentiment,
                key=record["id"],
                value=record,
            )
            logger.info(
                "sentiment_processed",
                ticker=record["ticker"],
                sentiment=record["sentiment"],
                confidence=record["confidence"],
                source=record["source"],
            )
        except Exception as exc:
            logger.error("sentiment_processing_error", error=str(exc), payload_id=payload.get("id"))

    def run(self) -> None:
        """Start consuming from both raw topics — blocks until stopped."""
        topics = [
            self.settings.kafka_topic_raw_social,
            self.settings.kafka_topic_raw_news,
        ]
        consumer = KafkaConsumerClient(topics=topics)
        for topic in topics:
            consumer.register_handler(topic, self.process_message)
        logger.info("sentiment_engine_starting", topics=topics)
        consumer.start()
