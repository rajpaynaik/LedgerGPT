from __future__ import annotations

"""
Reddit ingestion using PRAW.
Polls hot/new posts and comments from financial subreddits.
"""
import asyncio
from datetime import datetime, timezone
from typing import Generator

import praw
import structlog

from config import get_settings
from stream.kafka_producer import KafkaProducerClient

logger = structlog.get_logger(__name__)

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options", "SecurityAnalysis"]
POLL_INTERVAL_SEC = 60


class RedditIngester:
    """Polls Reddit subreddits and pushes posts/comments to Kafka."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.producer = KafkaProducerClient()
        self._reddit = praw.Reddit(
            client_id=self.settings.reddit_client_id,
            client_secret=self.settings.reddit_client_secret.get_secret_value(),
            user_agent=self.settings.reddit_user_agent,
        )

    # ── Post streaming ──────────────────────────────────────────────────────
    def _iter_posts(
        self,
        subreddit_name: str,
        limit: int = 25,
    ) -> Generator[dict, None, None]:
        subreddit = self._reddit.subreddit(subreddit_name)
        for submission in subreddit.new(limit=limit):
            yield {
                "source": "reddit",
                "subreddit": subreddit_name,
                "id": submission.id,
                "title": submission.title,
                "text": submission.selftext,
                "score": submission.score,
                "upvote_ratio": submission.upvote_ratio,
                "num_comments": submission.num_comments,
                "created_at": datetime.fromtimestamp(
                    submission.created_utc, tz=timezone.utc
                ).isoformat(),
                "url": submission.url,
            }

    def _iter_comments(
        self,
        subreddit_name: str,
        limit: int = 50,
    ) -> Generator[dict, None, None]:
        subreddit = self._reddit.subreddit(subreddit_name)
        for comment in subreddit.comments(limit=limit):
            yield {
                "source": "reddit_comment",
                "subreddit": subreddit_name,
                "id": comment.id,
                "text": comment.body,
                "score": comment.score,
                "created_at": datetime.fromtimestamp(
                    comment.created_utc, tz=timezone.utc
                ).isoformat(),
                "post_id": comment.link_id,
            }

    # ── Public API ─────────────────────────────────────────────────────────
    def fetch_posts(
        self,
        subreddits: list[str] | None = None,
        limit: int = 25,
    ) -> list[dict]:
        """Single-pass fetch for backtesting / backfill."""
        results = []
        for sub in (subreddits or SUBREDDITS):
            try:
                posts = list(self._iter_posts(sub, limit))
                results.extend(posts)
                logger.info("reddit_fetched", subreddit=sub, count=len(posts))
            except Exception as exc:
                logger.error("reddit_fetch_error", subreddit=sub, error=str(exc))
        return results

    async def stream_loop(self, subreddits: list[str] | None = None) -> None:
        """Continuous poll loop — runs until cancelled."""
        targets = subreddits or SUBREDDITS
        while True:
            for sub in targets:
                try:
                    for post in self._iter_posts(sub):
                        self.producer.publish(
                            topic=self.settings.kafka_topic_raw_social,
                            key=f"reddit:{post['id']}",
                            value=post,
                        )
                    for comment in self._iter_comments(sub, limit=25):
                        self.producer.publish(
                            topic=self.settings.kafka_topic_raw_social,
                            key=f"reddit_c:{comment['id']}",
                            value=comment,
                        )
                except Exception as exc:
                    logger.error("reddit_stream_error", subreddit=sub, error=str(exc))
            await asyncio.sleep(POLL_INTERVAL_SEC)
