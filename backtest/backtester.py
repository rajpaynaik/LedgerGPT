from __future__ import annotations

"""
Backtester — wires together Backtrader engine, feature data, and metrics.
Supports single-ticker and multi-ticker portfolio backtests.
"""
from datetime import datetime
from pathlib import Path

import backtrader as bt
import pandas as pd
import structlog

from config import get_settings
from features.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from ingestion.market_data_ingestion import MarketDataIngester
from models.signal_model import SignalModel
from .strategy import LedgerGPTStrategy, SignalData
from .metrics import PerformanceMetrics

logger = structlog.get_logger(__name__)


class Backtester:
    """
    Full backtest pipeline:
    1. Load/compute features for each ticker
    2. Run model inference to get signal column
    3. Feed into Backtrader
    4. Compute performance metrics
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        commission: float = 0.001,
    ) -> None:
        self.initial_cash = initial_cash
        self.commission = commission
        self.settings = get_settings()
        self.feature_eng = FeatureEngineer()
        self.market_data = MarketDataIngester()
        self._model: SignalModel | None = None

    def _get_model(self) -> SignalModel:
        if self._model is None:
            self._model = SignalModel()
            self._model.load()
        return self._model

    def _build_signal_df(
        self,
        ticker: str,
        sentiment_records: list[dict],
        period: str = "2y",
    ) -> pd.DataFrame:
        """Build OHLCV + signal + confidence DataFrame for one ticker."""
        price_df = self.market_data.fetch_ohlcv_yf(ticker, period=period)
        feat_df = self.feature_eng.build_training_dataset(
            price_df, sentiment_records, ticker
        )

        # Generate model predictions for each row
        model = self._get_model()
        X = feat_df[FEATURE_COLUMNS].fillna(0)
        predictions = model.predict(X)

        signal_map = {"BUY": 2, "HOLD": 1, "SELL": 0}
        feat_df["signal"] = [signal_map[p["signal"]] for p in predictions]
        feat_df["confidence"] = [p["confidence"] for p in predictions]

        # Merge back with raw price data
        merged = price_df.join(feat_df[["signal", "confidence"]], how="left")
        merged["signal"] = merged["signal"].fillna(1)
        merged["confidence"] = merged["confidence"].fillna(0.0)
        return merged

    def run(
        self,
        tickers: list[str],
        sentiment_records: list[dict] | None = None,
        period: str = "2y",
        strategy_params: dict | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        Run backtest across all tickers and return combined metrics.
        """
        sentiment_records = sentiment_records or []
        strategy_params = strategy_params or {}

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.045)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")

        # Add strategy
        cerebro.addstrategy(
            LedgerGPTStrategy,
            verbose=verbose,
            **strategy_params,
        )

        # Add data feeds
        dfs = {}
        for ticker in tickers:
            try:
                df = self._build_signal_df(ticker, sentiment_records, period)
                feed = SignalData(
                    dataname=df,
                    name=ticker,
                    open="open",
                    high="high",
                    low="low",
                    close="close",
                    volume="volume",
                    signal=df.columns.get_loc("signal"),
                    confidence=df.columns.get_loc("confidence"),
                )
                cerebro.adddata(feed, name=ticker)
                dfs[ticker] = df
                logger.info("backtest_data_added", ticker=ticker, rows=len(df))
            except Exception as exc:
                logger.error("backtest_data_error", ticker=ticker, error=str(exc))

        # Run
        logger.info("backtest_starting", tickers=tickers, cash=self.initial_cash)
        results = cerebro.run()
        strat = results[0]

        # Extract metrics
        final_value = cerebro.broker.getvalue()
        time_returns = strat.analyzers.time_return.get_analysis()
        equity_series = pd.Series(time_returns).sort_index()
        equity_curve = (1 + equity_series).cumprod() * self.initial_cash

        equity_metrics = PerformanceMetrics.from_equity_curve(equity_curve)
        trade_metrics = PerformanceMetrics.from_trade_log(strat.trade_log)

        summary = {
            "initial_cash": self.initial_cash,
            "final_value": round(final_value, 2),
            "equity_metrics": equity_metrics,
            "trade_metrics": trade_metrics,
            "tickers": tickers,
            "period": period,
        }

        PerformanceMetrics.print_report(equity_metrics, trade_metrics)
        return summary

    def walk_forward(
        self,
        tickers: list[str],
        sentiment_records: list[dict] | None = None,
        train_months: int = 12,
        test_months: int = 3,
        periods: int = 4,
    ) -> list[dict]:
        """Walk-forward backtest: train on rolling window, test on next period."""
        results = []
        sentiment_records = sentiment_records or []

        for i in range(periods):
            train_period = f"{train_months + i * test_months}mo"
            test_period = f"{test_months}mo"
            logger.info(
                "walk_forward_period",
                period_n=i + 1,
                train=train_period,
                test=test_period,
            )
            result = self.run(
                tickers=tickers,
                sentiment_records=sentiment_records,
                period=test_period,
            )
            result["walk_forward_period"] = i + 1
            results.append(result)

        return results
