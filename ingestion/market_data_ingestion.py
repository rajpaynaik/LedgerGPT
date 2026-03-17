from __future__ import annotations

"""
Real-time and historical market data ingestion.
Sources: Polygon.io (primary), Yahoo Finance (fallback), Alpaca.
"""
import asyncio
from datetime import datetime, date, timedelta
from typing import Any

import structlog
import yfinance as yf
import pandas as pd

from config import get_settings

logger = structlog.get_logger(__name__)

WATCHLIST = [
    "TSLA", "AAPL", "AMZN", "MSFT", "NVDA", "META",
    "GOOGL", "AMD", "BABA", "COIN", "SPY", "QQQ",
]


class MarketDataIngester:
    """
    Provides OHLCV bars, quotes, and real-time price snapshots.
    Falls back to Yahoo Finance when Polygon quota is exhausted.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._polygon_client = None
        self._alpaca_client = None
        self._init_clients()

    def _init_clients(self) -> None:
        polygon_key = self.settings.polygon_api_key.get_secret_value()
        if polygon_key:
            try:
                from polygon import RESTClient
                self._polygon_client = RESTClient(api_key=polygon_key)
                logger.info("polygon_client_initialised")
            except Exception as exc:
                logger.warning("polygon_init_failed", error=str(exc))

        alpaca_key = self.settings.alpaca_api_key.get_secret_value()
        if alpaca_key:
            try:
                from alpaca.data import StockHistoricalDataClient
                from alpaca.data.live import StockDataStream
                self._alpaca_data = StockHistoricalDataClient(
                    api_key=alpaca_key,
                    secret_key=self.settings.alpaca_secret_key.get_secret_value(),
                )
                logger.info("alpaca_client_initialised")
            except Exception as exc:
                logger.warning("alpaca_init_failed", error=str(exc))

    # ── Yahoo Finance (fallback) ────────────────────────────────────────────
    def fetch_ohlcv_yf(
        self,
        ticker: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download OHLCV bars from Yahoo Finance."""
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns=str.lower)
        df["ticker"] = ticker
        logger.debug("yf_ohlcv_fetched", ticker=ticker, rows=len(df))
        return df

    def fetch_multi_ohlcv_yf(
        self,
        tickers: list[str] | None = None,
        period: str = "3mo",
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        tickers = tickers or WATCHLIST
        result = {}
        for ticker in tickers:
            try:
                result[ticker] = self.fetch_ohlcv_yf(ticker, period, interval)
            except Exception as exc:
                logger.error("yf_fetch_error", ticker=ticker, error=str(exc))
        return result

    # ── Polygon ────────────────────────────────────────────────────────────
    def fetch_ohlcv_polygon(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        timespan: str = "day",
    ) -> pd.DataFrame:
        if not self._polygon_client:
            return self.fetch_ohlcv_yf(ticker)
        try:
            aggs = self._polygon_client.get_aggs(
                ticker,
                1,
                timespan,
                from_date,
                to_date,
                limit=50000,
            )
            rows = [
                {
                    "open": a.open, "high": a.high, "low": a.low,
                    "close": a.close, "volume": a.volume,
                    "vwap": a.vwap,
                    "timestamp": pd.to_datetime(a.timestamp, unit="ms", utc=True),
                }
                for a in aggs
            ]
            df = pd.DataFrame(rows).set_index("timestamp")
            df["ticker"] = ticker
            return df
        except Exception as exc:
            logger.error("polygon_fetch_error", ticker=ticker, error=str(exc))
            return self.fetch_ohlcv_yf(ticker)

    # ── Snapshot (latest price) ────────────────────────────────────────────
    def get_snapshot(self, tickers: list[str] | None = None) -> dict[str, dict]:
        tickers = tickers or WATCHLIST
        data = {}
        try:
            raw = yf.download(
                tickers,
                period="1d",
                interval="1m",
                progress=False,
                auto_adjust=True,
            )
            if hasattr(raw.columns, "levels"):
                for t in tickers:
                    try:
                        close = raw["Close"][t].dropna()
                        data[t] = {
                            "price": float(close.iloc[-1]),
                            "volume": float(raw["Volume"][t].iloc[-1]),
                            "timestamp": close.index[-1].isoformat(),
                        }
                    except Exception:
                        pass
        except Exception as exc:
            logger.error("snapshot_error", error=str(exc))
        return data

    # ── Streaming (Alpaca WebSocket) ───────────────────────────────────────
    async def stream_quotes(self, tickers: list[str] | None = None) -> None:
        """Stream real-time quotes via Alpaca WebSocket."""
        tickers = tickers or WATCHLIST
        alpaca_key = self.settings.alpaca_api_key.get_secret_value()
        if not alpaca_key:
            logger.warning("alpaca_key_missing_cannot_stream")
            return

        from alpaca.data.live import StockDataStream
        from alpaca.data.enums import DataFeed

        stream = StockDataStream(
            api_key=alpaca_key,
            secret_key=self.settings.alpaca_secret_key.get_secret_value(),
            feed=DataFeed.IEX,
        )

        async def on_quote(data: Any) -> None:
            logger.debug(
                "live_quote",
                ticker=data.symbol,
                bid=data.bid_price,
                ask=data.ask_price,
            )

        stream.subscribe_quotes(on_quote, *tickers)
        logger.info("alpaca_stream_starting", tickers=tickers)
        await stream.run()
