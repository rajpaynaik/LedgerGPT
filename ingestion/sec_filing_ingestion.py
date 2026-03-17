"""
SEC EDGAR filing ingestion — 10-K, 10-Q, 8-K.
Uses the free EDGAR full-text search API (no key required).
Publishes structured filing events to Kafka.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import aiohttp
import structlog

from config import get_settings
from stream.kafka_producer import KafkaProducerClient

logger = structlog.get_logger(__name__)

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q={query}&dateRange=custom&startdt={start}&enddt={end}&forms={form}"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EDGAR_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
EDGAR_FULL_TEXT_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms={form}&dateRange=custom&startdt={start}&enddt={end}"

HEADERS = {
    "User-Agent": "LedgerGPT research@ledgergpt.ai",
    "Accept-Encoding": "gzip, deflate",
}

# CIK mappings for watched tickers (extend as needed)
TICKER_CIK_MAP = {
    "TSLA": 1318605,
    "AAPL": 320193,
    "AMZN": 1018724,
    "MSFT": 789019,
    "NVDA": 1045810,
    "META": 1326801,
    "GOOGL": 1652044,
    "AMD": 2488,
    "COIN": 1679788,
}

FORM_IMPACT_MAP = {
    "8-K":  ("regulatory_event", "high"),
    "10-K": ("earnings_event", "medium"),
    "10-Q": ("earnings_event", "medium"),
    "SC 13G": ("ma_event", "low"),
    "SC 13D": ("ma_event", "high"),
    "S-1":  ("ma_event", "medium"),
    "DEF 14A": ("regulatory_event", "low"),
}

POLL_INTERVAL_SEC = 900  # 15 minutes


class SECFilingIngester:
    """
    Polls EDGAR for recent filings and publishes structured events to Kafka.
    Extracts event type (earnings, M&A, regulatory) from form type.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.producer = KafkaProducerClient()

    # ── Fetch recent filings for a CIK ──────────────────────────────────────
    async def _fetch_submissions(
        self, session: aiohttp.ClientSession, cik: int
    ) -> list[dict]:
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
        try:
            async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json(content_type=None)

            filings = data.get("filings", {}).get("recent", {})
            if not filings:
                return []

            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            primary_docs = filings.get("primaryDocument", [])
            descriptions = filings.get("primaryDocDescription", [])

            results = []
            for i, form in enumerate(forms):
                if form not in FORM_IMPACT_MAP:
                    continue
                event_type, urgency = FORM_IMPACT_MAP[form]
                results.append({
                    "cik": cik,
                    "form": form,
                    "filing_date": dates[i] if i < len(dates) else "",
                    "accession_number": accessions[i] if i < len(accessions) else "",
                    "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
                    "description": descriptions[i] if i < len(descriptions) else "",
                    "event_type": event_type,
                    "urgency": urgency,
                })
            return results[:20]  # most recent 20 qualifying filings
        except Exception as exc:
            logger.error("edgar_fetch_error", cik=cik, error=str(exc))
            return []

    # ── Fetch 8-K item descriptions (most important for events) ─────────────
    async def _fetch_8k_items(
        self,
        session: aiohttp.ClientSession,
        cik: int,
        accession: str,
    ) -> str:
        """Try to extract 8-K item text for better event classification."""
        clean_acc = accession.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{clean_acc}/{accession}.txt"
        )
        try:
            async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    text = await resp.text(errors="ignore")
                    # Extract just the first 2000 chars for LLM processing
                    return text[:2000].strip()
        except Exception:
            pass
        return ""

    # ── Build Kafka payload from a filing ────────────────────────────────────
    def _build_payload(self, ticker: str, filing: dict, raw_text: str = "") -> dict:
        return {
            "source": "sec_edgar",
            "id": f"edgar_{filing['cik']}_{filing['accession_number']}",
            "ticker": ticker,
            "title": f"{filing['form']} filing by {ticker}: {filing['description']}",
            "text": raw_text or f"{ticker} filed {filing['form']} on {filing['filing_date']}. {filing['description']}",
            "form_type": filing["form"],
            "filing_date": filing["filing_date"],
            "event_type": filing["event_type"],
            "urgency": filing["urgency"],
            "accession_number": filing["accession_number"],
            "published_at": filing["filing_date"],
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Public fetch (single pass, for backfill) ──────────────────────────────
    async def fetch_filings(
        self,
        tickers: list[str] | None = None,
    ) -> list[dict]:
        tickers = tickers or list(TICKER_CIK_MAP.keys())
        results = []
        async with aiohttp.ClientSession() as session:
            for ticker in tickers:
                cik = TICKER_CIK_MAP.get(ticker.upper())
                if not cik:
                    logger.warning("no_cik_for_ticker", ticker=ticker)
                    continue
                filings = await self._fetch_submissions(session, cik)
                for filing in filings:
                    payload = self._build_payload(ticker, filing)
                    results.append(payload)
                    logger.info(
                        "edgar_filing_fetched",
                        ticker=ticker,
                        form=filing["form"],
                        date=filing["filing_date"],
                    )
        return results

    # ── Continuous stream loop ────────────────────────────────────────────────
    async def stream_loop(self, tickers: list[str] | None = None) -> None:
        tickers = tickers or list(TICKER_CIK_MAP.keys())
        seen: set[str] = set()
        while True:
            filings = await self.fetch_filings(tickers)
            for payload in filings:
                filing_id = payload["id"]
                if filing_id in seen:
                    continue
                seen.add(filing_id)
                self.producer.publish(
                    topic=self.settings.kafka_topic_raw_news,
                    key=filing_id,
                    value=payload,
                )
            logger.info("edgar_cycle_complete", total=len(filings), new=len(seen))
            await asyncio.sleep(POLL_INTERVAL_SEC)
