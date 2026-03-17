"""
Earnings call transcript ingestion.
Sources: Seeking Alpha RSS (free), Motley Fool Transcripts RSS,
         and a fallback scraper for public SEC 8-K exhibits.
Publishes transcript chunks to Kafka for LLM processing.
"""
import asyncio
import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import aiohttp
import structlog

from config import get_settings
from stream.kafka_producer import KafkaProducerClient

logger = structlog.get_logger(__name__)

TRANSCRIPT_RSS_FEEDS = {
    "motley_fool_transcripts": "https://www.fool.com/feeds/rss/recent-articles.aspx",
    "seeking_alpha_transcripts": "https://seekingalpha.com/earnings/earnings-call-transcripts.xml",
}

# Chunk size to avoid exceeding LLM context limits
TRANSCRIPT_CHUNK_CHARS = 1500
POLL_INTERVAL_SEC = 600  # 10 minutes

EARNINGS_KEYWORDS = [
    "earnings per share", "revenue", "guidance", "outlook", "fiscal year",
    "quarterly results", "beat expectations", "miss expectations",
    "operating income", "gross margin", "year-over-year", "Q1", "Q2", "Q3", "Q4",
]


class EarningsTranscriptIngester:
    """
    Fetches earnings call transcripts from public RSS feeds.
    Chunks long transcripts and publishes each chunk to Kafka.
    Each chunk carries event metadata so the LLM can classify it.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.producer = KafkaProducerClient()

    def _extract_ticker_from_title(self, title: str) -> str | None:
        """Extract ticker from RSS title like 'TSLA Q1 2024 Earnings Call Transcript'."""
        match = re.search(r'\b([A-Z]{2,5})\b', title)
        return match.group(1) if match else None

    def _is_earnings_related(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in EARNINGS_KEYWORDS)

    def _chunk_text(self, text: str, max_chars: int = TRANSCRIPT_CHUNK_CHARS) -> list[str]:
        """Split text into sentence-aware chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) > max_chars:
                if current:
                    chunks.append(current.strip())
                current = sentence
            else:
                current += " " + sentence
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _fingerprint(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    # ── RSS parsing ──────────────────────────────────────────────────────────
    async def _fetch_rss(
        self,
        session: aiohttp.ClientSession,
        name: str,
        url: str,
    ) -> list[dict]:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return []
                text = await resp.text(errors="ignore")

            root = ET.fromstring(text)
            items = root.findall(".//item") or root.findall(".//entry")
            articles = []
            for item in items[:15]:
                title_el = item.find("title")
                desc_el = item.find("description") or item.find("summary")
                link_el = item.find("link")
                pub_el = item.find("pubDate") or item.find("published")

                title = title_el.text if title_el is not None else ""
                body = desc_el.text if desc_el is not None else ""
                link = link_el.text if link_el is not None else ""
                pub = pub_el.text if pub_el is not None else ""

                # Only process earnings-related content
                if not self._is_earnings_related(title + " " + body):
                    continue

                ticker = self._extract_ticker_from_title(title)
                articles.append({
                    "title": title,
                    "body": body or title,
                    "link": link,
                    "published_at": pub,
                    "ticker": ticker,
                    "feed": name,
                })
            return articles
        except Exception as exc:
            logger.error("transcript_rss_error", feed=name, error=str(exc))
            return []

    # ── Build Kafka payloads from an article ─────────────────────────────────
    def _build_payloads(self, article: dict) -> list[dict]:
        """Chunk transcript into LLM-sized pieces, one Kafka message each."""
        chunks = self._chunk_text(article["body"])
        payloads = []
        for i, chunk in enumerate(chunks):
            doc_id = f"transcript_{self._fingerprint(article['link'] + str(i))}"
            payloads.append({
                "source": "earnings_transcript",
                "id": doc_id,
                "ticker": article.get("ticker"),
                "title": article["title"],
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "published_at": article.get("published_at", ""),
                "url": article.get("link", ""),
                "event_type": "earnings_event",
                "urgency": "high",
                "processed_at": datetime.now(timezone.utc).isoformat(),
            })
        return payloads

    # ── Public fetch ──────────────────────────────────────────────────────────
    async def fetch_transcripts(self) -> list[dict]:
        all_payloads = []
        async with aiohttp.ClientSession() as session:
            for name, url in TRANSCRIPT_RSS_FEEDS.items():
                articles = await self._fetch_rss(session, name, url)
                for article in articles:
                    payloads = self._build_payloads(article)
                    all_payloads.extend(payloads)
        logger.info("transcripts_fetched", chunks=len(all_payloads))
        return all_payloads

    # ── Continuous loop ───────────────────────────────────────────────────────
    async def stream_loop(self) -> None:
        seen: set[str] = set()
        while True:
            payloads = await self.fetch_transcripts()
            new_count = 0
            for payload in payloads:
                if payload["id"] in seen:
                    continue
                seen.add(payload["id"])
                self.producer.publish(
                    topic=self.settings.kafka_topic_raw_news,
                    key=payload["id"],
                    value=payload,
                )
                new_count += 1
            logger.info("transcript_cycle_complete", new=new_count, total_seen=len(seen))
            await asyncio.sleep(POLL_INTERVAL_SEC)
