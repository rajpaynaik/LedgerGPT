from __future__ import annotations

"""
LedgerGPT backtrader strategy.
Uses pre-computed signal column from the feature DataFrame
so the exact same model artefact is replayed on historical data.
"""
import backtrader as bt
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

SIGNAL_BUY = 2
SIGNAL_HOLD = 1
SIGNAL_SELL = 0


class SignalData(bt.feeds.PandasData):
    """Extended Backtrader data feed with signal and confidence columns."""
    lines = ("signal", "confidence")
    params = (
        ("signal", -1),
        ("confidence", -1),
    )


class LedgerGPTStrategy(bt.Strategy):
    """
    Signal-driven strategy:
    - BUY when model signal == BUY and confidence >= min_confidence
    - SELL / EXIT when model signal == SELL and confidence >= min_confidence
    - Position sizing based on confidence and Kelly criterion approximation
    """

    params = (
        ("min_confidence", 0.60),
        ("max_position_pct", 0.20),  # max 20% of portfolio per position
        ("stop_loss_pct", 0.05),     # 5% stop loss
        ("take_profit_pct", 0.12),   # 12% take profit
        ("verbose", False),
    )

    def __init__(self) -> None:
        self.order_map: dict[str, bt.Order] = {}
        self.entry_price_map: dict[str, float] = {}
        self.trade_log: list[dict] = []

    def log(self, msg: str) -> None:
        if self.params.verbose:
            logger.info("backtest_strategy", date=self.datas[0].datetime.date(0), msg=msg)

    def notify_order(self, order: bt.Order) -> None:
        if order.status in [order.Submitted, order.Accepted]:
            return
        ticker = order.data._name
        if order.status == order.Completed:
            direction = "BUY" if order.isbuy() else "SELL"
            self.log(f"{direction} {ticker} @ {order.executed.price:.2f}")
            self.order_map.pop(ticker, None)
            if order.isbuy():
                self.entry_price_map[ticker] = order.executed.price
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order {order.status} for {ticker}")
            self.order_map.pop(ticker, None)

    def notify_trade(self, trade: bt.Trade) -> None:
        if trade.isclosed:
            pnl = trade.pnlcomm
            self.trade_log.append({
                "ticker": trade.data._name,
                "pnl": pnl,
                "pnl_pct": pnl / (trade.price * trade.size) if trade.price * trade.size != 0 else 0,
                "open_date": bt.num2date(trade.dtopen).isoformat(),
                "close_date": bt.num2date(trade.dtclose).isoformat(),
            })
            self.log(f"Trade closed PnL: {pnl:.2f}")

    def next(self) -> None:
        for data in self.datas:
            ticker = data._name
            if ticker in self.order_map:
                continue

            signal = int(data.signal[0]) if not np.isnan(data.signal[0]) else SIGNAL_HOLD
            confidence = float(data.confidence[0]) if not np.isnan(data.confidence[0]) else 0.0
            position = self.getposition(data)
            price = data.close[0]

            # Stop-loss / take-profit on open positions
            if position.size > 0:
                entry = self.entry_price_map.get(ticker, price)
                drawdown = (price - entry) / entry
                if drawdown <= -self.params.stop_loss_pct:
                    self.order_map[ticker] = self.sell(data=data)
                    self.log(f"STOP LOSS {ticker} drawdown={drawdown:.2%}")
                    continue
                if drawdown >= self.params.take_profit_pct:
                    self.order_map[ticker] = self.sell(data=data)
                    self.log(f"TAKE PROFIT {ticker} gain={drawdown:.2%}")
                    continue

            # Signal-based entry / exit
            if signal == SIGNAL_BUY and confidence >= self.params.min_confidence:
                if not position:
                    portfolio_value = self.broker.getvalue()
                    alloc = portfolio_value * self.params.max_position_pct * confidence
                    size = int(alloc / price)
                    if size > 0:
                        self.order_map[ticker] = self.buy(data=data, size=size)
                        self.log(f"BUY signal {ticker} conf={confidence:.2f} size={size}")

            elif signal == SIGNAL_SELL and confidence >= self.params.min_confidence:
                if position:
                    self.order_map[ticker] = self.sell(data=data)
                    self.log(f"SELL signal {ticker} conf={confidence:.2f}")

    def stop(self) -> None:
        logger.info(
            "strategy_finished",
            final_portfolio=self.broker.getvalue(),
            trades_closed=len(self.trade_log),
        )
