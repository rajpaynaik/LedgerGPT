"""
Start all data ingestion loops concurrently.
Includes: Twitter, Reddit, News, SEC EDGAR filings, Earnings transcripts.
Usage: python -m scripts.run_ingestion
"""
import asyncio
import threading
import structlog

from ingestion.reddit_ingestion import RedditIngester
from ingestion.news_ingestion import NewsIngester
from ingestion.twitter_ingestion import TwitterIngester
from ingestion.sec_filing_ingestion import SECFilingIngester
from ingestion.earnings_transcript_ingestion import EarningsTranscriptIngester

logger = structlog.get_logger(__name__)


async def main() -> None:
    logger.info("ingestion_workers_starting")

    # Async loops
    reddit      = RedditIngester()
    news        = NewsIngester()
    sec         = SECFilingIngester()
    transcripts = EarningsTranscriptIngester()

    tasks = [
        asyncio.create_task(reddit.stream_loop(),      name="reddit"),
        asyncio.create_task(news.stream_loop(),        name="news"),
        asyncio.create_task(sec.stream_loop(),         name="sec_edgar"),
        asyncio.create_task(transcripts.stream_loop(), name="earnings_transcripts"),
    ]

    # Twitter uses a blocking streaming client — run in a daemon thread
    twitter = TwitterIngester()
    twitter_thread = threading.Thread(target=twitter.start_stream, daemon=True)
    twitter_thread.start()
    logger.info("twitter_stream_thread_started")

    logger.info(
        "all_ingestion_tasks_started",
        async_tasks=[t.get_name() for t in tasks],
    )
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
