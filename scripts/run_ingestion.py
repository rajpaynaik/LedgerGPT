"""
Start all data ingestion loops concurrently.
Usage: python -m scripts.run_ingestion
"""
import asyncio
import structlog

from ingestion.reddit_ingestion import RedditIngester
from ingestion.news_ingestion import NewsIngester
from ingestion.twitter_ingestion import TwitterIngester

logger = structlog.get_logger(__name__)


async def main() -> None:
    logger.info("ingestion_workers_starting")
    reddit = RedditIngester()
    news = NewsIngester()

    tasks = [
        asyncio.create_task(reddit.stream_loop(), name="reddit"),
        asyncio.create_task(news.stream_loop(), name="news"),
    ]

    # Twitter uses a sync streaming client in a thread
    import threading
    twitter = TwitterIngester()
    twitter_thread = threading.Thread(target=twitter.start_stream, daemon=True)
    twitter_thread.start()
    logger.info("twitter_stream_thread_started")

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
