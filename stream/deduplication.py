"""
Redis-backed deduplication filter.
Uses a sliding-window bloom filter via a sorted-set TTL pattern.
"""
import hashlib
import time
from typing import Any

import redis
import structlog

from config import get_settings

logger = structlog.get_logger(__name__)

DEFAULT_TTL_SEC = 3600  # 1 hour dedup window
KEY_PREFIX = "dedup:"


class DeduplicationFilter:
    """
    Deduplicates raw social / news messages using Redis.
    Returns True if the message is new, False if already seen.
    """

    def __init__(self, ttl_sec: int = DEFAULT_TTL_SEC) -> None:
        self.ttl = ttl_sec
        settings = get_settings()
        self._redis = redis.from_url(settings.redis_url, decode_responses=True)

    def _fingerprint(self, payload: dict) -> str:
        """SHA-256 of source + id (fast, low collision)."""
        source = payload.get("source", "unknown")
        doc_id = payload.get("id", "")
        raw = f"{source}:{doc_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def is_new(self, payload: dict) -> bool:
        """Return True if payload has not been seen within the TTL window."""
        fp = self._fingerprint(payload)
        key = f"{KEY_PREFIX}{fp}"
        now = int(time.time())

        # SET NX EX — atomic check-and-set
        result = self._redis.set(key, now, nx=True, ex=self.ttl)
        is_new = result is True
        if not is_new:
            logger.debug("dedup_filtered", fingerprint=fp, source=payload.get("source"))
        return is_new

    def mark_seen(self, payload: dict) -> None:
        """Explicitly mark a payload as seen (call after processing)."""
        fp = self._fingerprint(payload)
        key = f"{KEY_PREFIX}{fp}"
        self._redis.set(key, int(time.time()), ex=self.ttl)

    def filter_batch(self, payloads: list[dict]) -> list[dict]:
        """Return only unseen payloads from a list."""
        return [p for p in payloads if self.is_new(p)]

    def stats(self) -> dict:
        count = len(self._redis.keys(f"{KEY_PREFIX}*"))
        return {"dedup_keys_active": count, "ttl_sec": self.ttl}
