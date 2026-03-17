from __future__ import annotations

"""
Order Manager — converts ML signals into broker orders
with position sizing, risk checks, and execution logging.
"""
from datetime import datetime, timezone

import structlog

from config import get_settings
from .broker_api import BrokerAPI, BaseBroker

logger = structlog.get_logger(__name__)

MIN_CONFIDENCE_TO_TRADE = 0.65
MIN_POSITION_USD = 100.0


class OrderManager:
    """
    Translates signal dicts → broker orders with risk management.

    Risk rules applied before any order:
    - Confidence must exceed MIN_CONFIDENCE_TO_TRADE
    - Position size capped at max_position_size_usd (from settings)
    - Risk per trade capped at risk_per_trade_pct of portfolio
    - No trading outside market hours (TODO: integrate schedule)
    """

    def __init__(self, broker: BaseBroker | None = None) -> None:
        self.settings = get_settings()
        self._broker = broker or BrokerAPI.create()
        self._order_log: list[dict] = []

    def _position_size(
        self,
        price: float,
        confidence: float,
        portfolio_value: float,
    ) -> float:
        """
        Kelly-fraction position sizing.
        size = (portfolio * risk_pct * confidence) / price
        """
        risk_amount = portfolio_value * self.settings.risk_per_trade_pct * confidence
        size = min(risk_amount, self.settings.max_position_size_usd) / price
        return max(0, round(size, 4))

    def _risk_check(self, signal: dict, account: dict) -> tuple[bool, str]:
        """Return (approved, reason)."""
        if signal["confidence"] < MIN_CONFIDENCE_TO_TRADE:
            return False, f"confidence {signal['confidence']:.2f} below threshold {MIN_CONFIDENCE_TO_TRADE}"
        if signal["signal"] == "HOLD":
            return False, "HOLD signal — no action"
        if account["buying_power"] < MIN_POSITION_USD:
            return False, "insufficient buying power"
        return True, "ok"

    def execute_signal(self, signal: dict, current_price: float) -> dict | None:
        """
        Execute a single signal. Returns order dict or None if skipped.
        """
        account = self._broker.get_account()
        approved, reason = self._risk_check(signal, account)

        if not approved:
            logger.info(
                "order_skipped",
                ticker=signal["ticker"],
                reason=reason,
                signal=signal["signal"],
            )
            return None

        portfolio_value = account["portfolio_value"]
        qty = self._position_size(current_price, signal["confidence"], portfolio_value)

        if qty <= 0:
            logger.info("order_skipped", ticker=signal["ticker"], reason="qty=0")
            return None

        try:
            order = self._broker.place_order(
                ticker=signal["ticker"],
                qty=qty,
                side=signal["signal"],  # BUY or SELL
            )
            log_entry = {
                **order,
                "signal_confidence": signal["confidence"],
                "signal_reason": signal.get("reason", ""),
                "price_at_signal": current_price,
                "executed_at": datetime.now(timezone.utc).isoformat(),
            }
            self._order_log.append(log_entry)
            return log_entry
        except Exception as exc:
            logger.error(
                "order_execution_failed",
                ticker=signal["ticker"],
                error=str(exc),
            )
            return None

    def execute_batch(
        self,
        signals: list[dict],
        prices: dict[str, float],
    ) -> list[dict]:
        """Execute a list of signals, returning executed orders."""
        executed = []
        for signal in signals:
            ticker = signal["ticker"]
            price = prices.get(ticker)
            if price is None:
                logger.warning("no_price_for_ticker", ticker=ticker)
                continue
            result = self.execute_signal(signal, price)
            if result:
                executed.append(result)
        return executed

    @property
    def order_history(self) -> list[dict]:
        return list(self._order_log)
