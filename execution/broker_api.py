from __future__ import annotations

"""
Broker abstraction layer.
Currently implements Alpaca Markets (paper + live).
Easily extendable to IBKR, Tradier, etc.
"""
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import structlog

from config import get_settings

logger = structlog.get_logger(__name__)


class BaseBroker(ABC):
    @abstractmethod
    def place_order(
        self, ticker: str, qty: float, side: str, order_type: str = "market"
    ) -> dict:
        ...

    @abstractmethod
    def get_positions(self) -> list[dict]:
        ...

    @abstractmethod
    def get_account(self) -> dict:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...


class AlpacaBroker(BaseBroker):
    """Alpaca Markets broker implementation (paper/live)."""

    def __init__(self) -> None:
        settings = get_settings()
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            self._client = TradingClient(
                api_key=settings.alpaca_api_key.get_secret_value(),
                secret_key=settings.alpaca_secret_key.get_secret_value(),
                paper=settings.paper_trading,
            )
            self._OrderSide = OrderSide
            self._TimeInForce = TimeInForce
            self._MarketOrderRequest = MarketOrderRequest
            self._LimitOrderRequest = LimitOrderRequest
            logger.info(
                "alpaca_broker_initialised",
                paper=settings.paper_trading,
                base_url=settings.alpaca_base_url,
            )
        except ImportError:
            raise RuntimeError("alpaca-py not installed. Run: pip install alpaca-py")

    def place_order(
        self,
        ticker: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> dict:
        order_side = (
            self._OrderSide.BUY if side.upper() == "BUY" else self._OrderSide.SELL
        )
        if order_type == "market":
            req = self._MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=self._TimeInForce.DAY,
            )
        else:
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")
            req = self._LimitOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=self._TimeInForce.GTC,
                limit_price=limit_price,
            )
        order = self._client.submit_order(req)
        result = {
            "order_id": str(order.id),
            "ticker": ticker,
            "qty": qty,
            "side": side,
            "status": str(order.status),
            "submitted_at": str(order.submitted_at),
        }
        logger.info("order_placed", **result)
        return result

    def get_positions(self) -> list[dict]:
        positions = self._client.get_all_positions()
        return [
            {
                "ticker": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealised_pl": float(p.unrealized_pl),
                "unrealised_pl_pct": float(p.unrealized_plpc),
            }
            for p in positions
        ]

    def get_account(self) -> dict:
        acct = self._client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "daytrade_count": acct.daytrade_count,
        }

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception as exc:
            logger.error("cancel_order_failed", order_id=order_id, error=str(exc))
            return False

    def cancel_all_orders(self) -> int:
        cancelled = self._client.cancel_orders()
        return len(cancelled)


class BrokerAPI:
    """Factory that returns the configured broker."""

    @staticmethod
    def create() -> BaseBroker:
        settings = get_settings()
        if settings.broker.lower() == "alpaca":
            return AlpacaBroker()
        raise ValueError(f"Unsupported broker: {settings.broker}")
