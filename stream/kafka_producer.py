from __future__ import annotations

"""
Thin wrapper around confluent-kafka Producer with JSON serialisation,
retry logic, and dead-letter queue support.
"""
import json
import time
from typing import Any

import structlog
from confluent_kafka import Producer, KafkaError
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

logger = structlog.get_logger(__name__)


class KafkaProducerClient:
    """Thread-safe Kafka producer — use one instance per process."""

    def __init__(self) -> None:
        settings = get_settings()
        self._producer = Producer(
            {
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "acks": "all",
                "retries": 5,
                "retry.backoff.ms": 500,
                "compression.type": "lz4",
                "batch.size": 65536,
                "linger.ms": 20,
            }
        )
        self._dlq_topic = "dlq.failed"

    def _delivery_report(self, err: KafkaError | None, msg: Any) -> None:
        if err:
            logger.error(
                "kafka_delivery_failed",
                topic=msg.topic(),
                key=msg.key(),
                error=str(err),
            )
        else:
            logger.debug(
                "kafka_delivered",
                topic=msg.topic(),
                partition=msg.partition(),
                offset=msg.offset(),
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def publish(self, topic: str, value: dict, key: str = "") -> None:
        """Serialise payload to JSON and produce to Kafka."""
        try:
            self._producer.produce(
                topic=topic,
                key=key.encode("utf-8") if key else None,
                value=json.dumps(value, default=str).encode("utf-8"),
                callback=self._delivery_report,
            )
            self._producer.poll(0)
        except BufferError:
            # Queue full — flush and retry
            self._producer.flush(timeout=5)
            self._producer.produce(
                topic=topic,
                key=key.encode("utf-8") if key else None,
                value=json.dumps(value, default=str).encode("utf-8"),
                callback=self._delivery_report,
            )

    def send_to_dlq(self, original_topic: str, value: dict, reason: str) -> None:
        payload = {"original_topic": original_topic, "reason": reason, "data": value}
        self.publish(self._dlq_topic, payload)

    def flush(self, timeout: float = 30.0) -> None:
        self._producer.flush(timeout=timeout)

    def __del__(self) -> None:
        try:
            self._producer.flush(timeout=5)
        except Exception:
            pass
