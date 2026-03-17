"""
Financial news ingestion from NewsAPI and RSS feeds.
Publishes cleaned articles to Kafka.
"""
import asyncio
from datetime import datetime, timezone
from typing import Any

import aiohttp
import structlog
from newsapi import NewsApiClient

from config import get_settings
from stream.kafka_producer import KafkaProducerClient

logger = structlog.get_logger(__name__)

NEWS_QUERIES = [
    "stock market earnings",
    "Federal Reserve interest rates",
    "Tesla earnings",
    "Apple stock",
    "NVIDIA GPU AI",
    "SEC filing",
    "IPO",
    "merger acquisition",
]

RSS_FEEDS = {
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    "cnbc_top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "marketwatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "seeking_alpha": "https://seekingalpha.com/market_currents.xml",
}

POLL_INTERVAL_SEC = 300  # 5 minutes


class NewsIngester:
    """Polls NewsAPI and RSS feeds, publishes to Kafka."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.producer = KafkaProducerClient()
        self._newsapi = NewsApiClient(
            api_key=self.settings.newsapi_key.get_secret_value()
        )

    # ── NewsAPI ─────────────────────────────────────────────────────────────
    def fetch_newsapi(
        self,
        query: str = "stock market",
        page_size: int = 50,
    ) -> list[dict]:
        try:
            response = self._newsapi.get_everything(
                q=query,
                language="en",
                sort_by="publishedAt",
                page_size=page_size,
            )
            articles = []
            for art in response.get("articles", []):
                articles.append({
                    "source": "newsapi",
                    "id": art.get("url", ""),
                    "title": art.get("title", ""),
                    "text": art.get("description", "") or art.get("content", ""),
                    "url": art.get("url", ""),
                    "published_at": art.get("publishedAt", ""),
                    "source_name": art.get("source", {}).get("name", ""),
                })
            logger.info("newsapi_fetched", query=query, count=len(articles))
            return articles
        except Exception as exc:
            logger.error("newsapi_error", error=str(exc))
            return []

    # ── RSS feeds ──────────────────────────────────────────────────────────
    async def _fetch_rss(self, session: aiohttp.ClientSession, name: str, url: str) -> list[dict]:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                text = await resp.text()
            # Minimal RSS parsing without feedparser dependency
            import xml.etree.ElementTree as ET
            root = ET.fromstring(text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            items = root.findall(".//item") or root.findall(".//atom:entry", ns)
            articles = []
            for item in items[:20]:
                title_el = item.find("title")
                desc_el = item.find("description") or item.find("summary")
                link_el = item.find("link")
                pub_el = item.find("pubDate") or item.find("published")
                articles.append({
                    "source": f"rss_{name}",
                    "id": link_el.text if link_el is not None else "",
                    "title": title_el.text if title_el is not None else "",
                    "text": desc_el.text if desc_el is not None else "",
                    "url": link_el.text if link_el is not None else "",
                    "published_at": pub_el.text if pub_el is not None else "",
                })
            logger.debug("rss_fetched", feed=name, count=len(articles))
            return articles
        except Exception as exc:
            logger.error("rss_fetch_error", feed=name, error=str(exc))
            return []

    async def fetch_all_rss(self) -> list[dict]:
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_rss(session, name, url)
                for name, url in RSS_FEEDS.items()
            ]
            results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

    # ── Continuous loop ─────────────────────────────────────────────────────
    async def stream_loop(self) -> None:
        while True:
            # NewsAPI
            for query in NEWS_QUERIES:
                articles = self.fetch_newsapi(query=query, page_size=20)
                for art in articles:
                    self.producer.publish(
                        topic=self.settings.kafka_topic_raw_news,
                        key=art["id"],
                        value=art,
                    )
            # RSS
            rss_articles = await self.fetch_all_rss()
            for art in rss_articles:
                self.producer.publish(
                    topic=self.settings.kafka_topic_raw_news,
                    key=art["id"],
                    value=art,
                )
            logger.info("news_cycle_complete", total=len(rss_articles))
            await asyncio.sleep(POLL_INTERVAL_SEC)
