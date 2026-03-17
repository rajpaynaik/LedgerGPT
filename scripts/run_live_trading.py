"""
Live trading loop — generates signals, applies portfolio risk controls,
and executes approved orders via the broker API.

Usage: python -m scripts.run_live_trading [--dry-run]
Paper trading is the default (PAPER_TRADING=true in .env).
"""
import argparse
import asyncio
import time
from datetime import datetime, timezone

import structlog

from config import get_settings
from execution.order_manager import OrderManager
from execution.portfolio_manager import PortfolioManager, PortfolioRiskBreaker
from ingestion.market_data_ingestion import MarketDataIngester, WATCHLIST
from models.predict import SignalPredictor
from monitoring.alerting import AlertManager

logger = structlog.get_logger(__name__)
LOOP_INTERVAL_SEC = 300  # 5 minutes between signal sweeps


async def trading_loop(dry_run: bool = False) -> None:
    settings = get_settings()
    predictor      = SignalPredictor()
    market_data    = MarketDataIngester()
    order_mgr      = OrderManager()
    portfolio_mgr  = PortfolioManager()
    alert_mgr      = AlertManager()

    logger.info(
        "live_trading_starting",
        paper=settings.paper_trading,
        dry_run=dry_run,
        watchlist=WATCHLIST,
    )

    while True:
        try:
            # ── Generate signals ──────────────────────────────────────────
            signals = predictor.predict_all()
            alert_mgr.check_low_confidence(signals)

            # ── Get current prices ────────────────────────────────────────
            snapshot = market_data.get_snapshot()

            # ── Portfolio risk filter ─────────────────────────────────────
            try:
                approved = portfolio_mgr.approve_signals(signals)
            except PortfolioRiskBreaker as e:
                logger.critical("circuit_breaker_tripped", reason=str(e))
                alert_mgr._dispatch(
                    type("Alert", (), {
                        "name": "circuit_breaker", "severity": "critical",
                        "message": str(e), "context": {}
                    })()
                )
                await asyncio.sleep(LOOP_INTERVAL_SEC * 6)  # pause 30 min
                continue

            logger.info(
                "signals_approved",
                total=len(signals),
                approved=len(approved),
            )

            # ── Execute orders ────────────────────────────────────────────
            if not dry_run:
                executed = order_mgr.execute_batch(approved, snapshot)
                logger.info("orders_executed", count=len(executed))
            else:
                logger.info("dry_run_no_orders", approved_signals=[
                    {"ticker": s["ticker"], "signal": s["signal"], "confidence": s["confidence"]}
                    for s in approved
                ])

            # ── Portfolio summary ─────────────────────────────────────────
            summary = portfolio_mgr.portfolio_summary()
            logger.info(
                "portfolio_snapshot",
                equity=summary["equity"],
                open_positions=summary["open_positions"],
                unrealised_pl=summary["total_unrealised_pl"],
            )

        except Exception as exc:
            logger.error("trading_loop_error", error=str(exc))

        await asyncio.sleep(LOOP_INTERVAL_SEC)


def main() -> None:
    parser = argparse.ArgumentParser(description="LedgerGPT live trading")
    parser.add_argument("--dry-run", action="store_true", help="Generate signals but do not place orders")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("dry_run_mode_enabled")

    asyncio.run(trading_loop(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
