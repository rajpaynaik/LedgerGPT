from __future__ import annotations

"""
Generic Kafka consumer with JSON deserialisation, offset commit,
and pluggable message handler pattern.
"""
import json
import signal
import threading
from collections.abc import Callable
from typing import Any

import structlog
from confluent_kafka import Consumer, KafkaError, KafkaException, Message

from config import get_settings

logger = structlog.get_logger(__name__)

MessageHandler = Callable[[dict, str], None]


class KafkaConsumerClient:
    """
    Subscribes to one or more Kafka topics and dispatches messages
    to registered handlers.

    Usage:
        consumer = KafkaConsumerClient(topics=["raw.social"])
        consumer.register_handler("raw.social", my_handler_fn)
        consumer.start()   # blocks
    """

    def __init__(
        self,
        topics: list[str],
        group_id: str | None = None,
        auto_offset_reset: str = "latest",
    ) -> None:
        settings = get_settings()
        self._topics = topics
        self._handlers: dict[str, list[MessageHandler]] = {t: [] for t in topics}
        self._running = False
        self._consumer = Consumer(
            {
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "group.id": group_id or settings.kafka_group_id,
                "auto.offset.reset": auto_offset_reset,
                "enable.auto.commit": False,
                "max.poll.interval.ms": 300_000,
                "session.timeout.ms": 30_000,
            }
        )

    def register_handler(self, topic: str, handler: MessageHandler) -> None:
        if topic not in self._handlers:
            self._handlers[topic] = []
        self._handlers[topic].append(handler)

    def _process_message(self, msg: Message) -> None:
        topic = msg.topic()
        try:
            value = json.loads(msg.value().decode("utf-8"))
            key = msg.key().decode("utf-8") if msg.key() else ""
        except Exception as exc:
            logger.error("kafka_deserialise_error", topic=topic, error=str(exc))
            return

        for handler in self._handlers.get(topic, []):
            try:
                handler(value, key)
            except Exception as exc:
                logger.error(
                    "kafka_handler_error",
                    topic=topic,
                    handler=handler.__name__,
                    error=str(exc),
                )

    def start(self) -> None:
        """Blocking consume loop — gracefully handles SIGTERM/SIGINT."""
        self._consumer.subscribe(self._topics)
        self._running = True

        def _stop(signum, frame):
            logger.info("kafka_consumer_stopping")
            self._running = False

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)

        logger.info("kafka_consumer_started", topics=self._topics)
        try:
            while self._running:
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise KafkaException(msg.error())
                self._process_message(msg)
                self._consumer.commit(asynchronous=True)
        finally:
            self._consumer.close()
            logger.info("kafka_consumer_closed")

    def start_background(self) -> threading.Thread:
        """Start the consumer in a daemon thread."""
        t = threading.Thread(target=self.start, daemon=True)
        t.start()
        return t
