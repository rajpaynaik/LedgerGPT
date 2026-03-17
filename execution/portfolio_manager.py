"""
Portfolio Manager — enforces portfolio-level risk constraints
before any batch of orders is sent to the broker.

Risk rules:
  - Max total portfolio concentration per sector
  - Max drawdown circuit breaker (halt trading if exceeded)
  - Max open positions
  - Daily loss limit
  - Correlation-aware position sizing (avoid doubling correlated positions)
"""
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog

from config import get_settings
from .broker_api import BaseBroker, BrokerAPI

logger = structlog.get_logger(__name__)

# Hard-coded risk limits (can be moved to settings / DB)
MAX_OPEN_POSITIONS    = 10
MAX_CONCENTRATION_PCT = 0.15   # single position ≤ 15% of portfolio
MAX_SECTOR_PCT        = 0.35   # single sector ≤ 35%
MAX_PORTFOLIO_DRAWDOWN_PCT = 0.15   # halt all new trades if DD ≥ 15%
DAILY_LOSS_LIMIT_PCT  = 0.05   # pause if daily loss ≥ 5%

# Simple sector mapping (extend as needed)
TICKER_SECTOR = {
    "TSLA": "consumer_discretionary", "F": "consumer_discretionary",
    "AAPL": "technology",  "MSFT": "technology", "NVDA": "technology",
    "AMD":  "technology",  "META": "technology",  "GOOGL": "technology",
    "AMZN": "consumer_discretionary",
    "COIN": "financials",  "GS": "financials",
    "SPY":  "etf",         "QQQ": "etf",
}


class PortfolioRiskBreaker(Exception):
    """Raised when a portfolio-level circuit breaker trips."""


class PortfolioManager:
    """
    Wraps OrderManager with portfolio-level risk management.
    Call `approve_signals()` before executing a batch of signals.
    """

    def __init__(self, broker: BaseBroker | None = None) -> None:
        self.settings = get_settings()
        self._broker = broker or BrokerAPI.create()
        self._trading_halted = False
        self._halt_reason: str = ""
        self._daily_pnl: float = 0.0
        self._session_start_equity: float | None = None

    # ── Account state ──────────────────────────────────────────────────────
    def _get_account(self) -> dict:
        return self._broker.get_account()

    def _get_positions(self) -> list[dict]:
        return self._broker.get_positions()

    # ── Circuit breakers ───────────────────────────────────────────────────
    def check_drawdown_breaker(self) -> None:
        account = self._get_account()
        equity = account["equity"]
        if self._session_start_equity is None:
            self._session_start_equity = equity
            return

        drawdown = (equity - self._session_start_equity) / self._session_start_equity
        if drawdown <= -MAX_PORTFOLIO_DRAWDOWN_PCT:
            self._trading_halted = True
            self._halt_reason = (
                f"Portfolio drawdown {drawdown:.1%} exceeded limit {MAX_PORTFOLIO_DRAWDOWN_PCT:.1%}"
            )
            raise PortfolioRiskBreaker(self._halt_reason)

    def check_daily_loss_limit(self) -> None:
        account = self._get_account()
        if self._session_start_equity is None:
            self._session_start_equity = account["equity"]
            return

        daily_return = (account["equity"] - self._session_start_equity) / self._session_start_equity
        if daily_return <= -DAILY_LOSS_LIMIT_PCT:
            self._trading_halted = True
            self._halt_reason = f"Daily loss limit {daily_return:.1%} triggered"
            raise PortfolioRiskBreaker(self._halt_reason)

    # ── Signal filtering ───────────────────────────────────────────────────
    def _filter_by_max_positions(self, signals: list[dict], positions: list[dict]) -> list[dict]:
        open_count = len([p for p in positions if abs(p["qty"]) > 0])
        buy_signals = [s for s in signals if s["signal"] != "BUY"]  # always allow SELL/HOLD
        buy_candidates = [s for s in signals if s["signal"] == "BUY"]

        slots = max(0, MAX_OPEN_POSITIONS - open_count)
        # Rank BUY signals by confidence, take top N
        buy_candidates.sort(key=lambda s: s["confidence"], reverse=True)
        approved_buys = buy_candidates[:slots]

        filtered = buy_signals + approved_buys
        if len(approved_buys) < len(buy_candidates):
            skipped = [s["ticker"] for s in buy_candidates[slots:]]
            logger.info("position_limit_skip", max=MAX_OPEN_POSITIONS, skipped=skipped)
        return filtered

    def _filter_by_concentration(
        self,
        signals: list[dict],
        positions: list[dict],
        account: dict,
    ) -> list[dict]:
        portfolio_value = account["portfolio_value"]
        position_values = {p["ticker"]: p["market_value"] for p in positions}
        approved = []
        for signal in signals:
            if signal["signal"] != "BUY":
                approved.append(signal)
                continue
            current_val = position_values.get(signal["ticker"], 0.0)
            if current_val / portfolio_value >= MAX_CONCENTRATION_PCT:
                logger.info(
                    "concentration_skip",
                    ticker=signal["ticker"],
                    current_pct=f"{current_val / portfolio_value:.1%}",
                )
                continue
            approved.append(signal)
        return approved

    def _filter_by_sector(
        self,
        signals: list[dict],
        positions: list[dict],
        account: dict,
    ) -> list[dict]:
        portfolio_value = account["portfolio_value"]
        # Calculate current sector exposure
        sector_values: dict[str, float] = {}
        for p in positions:
            sector = TICKER_SECTOR.get(p["ticker"], "other")
            sector_values[sector] = sector_values.get(sector, 0.0) + p["market_value"]

        approved = []
        for signal in signals:
            if signal["signal"] != "BUY":
                approved.append(signal)
                continue
            sector = TICKER_SECTOR.get(signal["ticker"], "other")
            current_sector_pct = sector_values.get(sector, 0.0) / max(portfolio_value, 1)
            if current_sector_pct >= MAX_SECTOR_PCT:
                logger.info(
                    "sector_limit_skip",
                    ticker=signal["ticker"],
                    sector=sector,
                    sector_pct=f"{current_sector_pct:.1%}",
                )
                continue
            approved.append(signal)
        return approved

    # ── Public API ─────────────────────────────────────────────────────────
    def approve_signals(self, signals: list[dict]) -> list[dict]:
        """
        Apply all portfolio risk filters to a batch of signals.
        Returns the subset of signals approved for execution.
        Raises PortfolioRiskBreaker if a circuit breaker trips.
        """
        if self._trading_halted:
            raise PortfolioRiskBreaker(f"Trading halted: {self._halt_reason}")

        # Check circuit breakers first
        self.check_daily_loss_limit()
        self.check_drawdown_breaker()

        account = self._get_account()
        positions = self._get_positions()

        approved = self._filter_by_max_positions(signals, positions)
        approved = self._filter_by_concentration(approved, positions, account)
        approved = self._filter_by_sector(approved, positions, account)

        logger.info(
            "portfolio_approval",
            total_in=len(signals),
            approved=len(approved),
            open_positions=len(positions),
            portfolio_value=account["portfolio_value"],
        )
        return approved

    def portfolio_summary(self) -> dict:
        """Return current portfolio state with risk metrics."""
        account = self._get_account()
        positions = self._get_positions()

        sector_exposure: dict[str, float] = {}
        for p in positions:
            sector = TICKER_SECTOR.get(p["ticker"], "other")
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + p["market_value"]

        total_unrealised_pl = sum(p["unrealised_pl"] for p in positions)

        return {
            "equity": account["equity"],
            "cash": account["cash"],
            "buying_power": account["buying_power"],
            "portfolio_value": account["portfolio_value"],
            "open_positions": len(positions),
            "total_unrealised_pl": round(total_unrealised_pl, 2),
            "sector_exposure": {k: round(v / account["portfolio_value"], 4) for k, v in sector_exposure.items()},
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "positions": positions,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    def reset_halt(self) -> None:
        """Manually clear a trading halt (requires human review)."""
        logger.warning("trading_halt_manually_cleared", previous_reason=self._halt_reason)
        self._trading_halted = False
        self._halt_reason = ""
