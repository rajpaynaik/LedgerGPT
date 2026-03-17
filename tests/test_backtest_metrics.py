"""
Unit tests for PerformanceMetrics.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.metrics import PerformanceMetrics


def make_equity_curve(
    n: int = 252,
    total_return: float = 0.20,
    volatility: float = 0.15,
    seed: int = 42,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    daily_vol = volatility / np.sqrt(252)
    drift = (total_return / 252) - 0.5 * daily_vol**2
    returns = drift + daily_vol * rng.standard_normal(n)
    prices = 100_000 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx)


class TestPerformanceMetrics:
    def test_equity_metrics_keys(self):
        eq = make_equity_curve()
        metrics = PerformanceMetrics.from_equity_curve(eq)
        required_keys = [
            "total_return_pct", "annualised_return_pct", "annualised_volatility_pct",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown_pct", "max_drawdown_duration_days",
            "var_95_pct", "cvar_95_pct",
        ]
        for k in required_keys:
            assert k in metrics, f"Missing key: {k}"

    def test_positive_return_positive_sharpe(self):
        eq = make_equity_curve(total_return=0.30, volatility=0.10)
        metrics = PerformanceMetrics.from_equity_curve(eq)
        assert metrics["total_return_pct"] > 0
        assert metrics["sharpe_ratio"] > 0

    def test_max_drawdown_negative(self):
        eq = make_equity_curve()
        metrics = PerformanceMetrics.from_equity_curve(eq)
        assert metrics["max_drawdown_pct"] <= 0

    def test_cvar_less_than_var(self):
        eq = make_equity_curve()
        metrics = PerformanceMetrics.from_equity_curve(eq)
        # CVaR should be more negative (worse) than VaR
        assert metrics["cvar_95_pct"] <= metrics["var_95_pct"]

    def test_trade_metrics_empty(self):
        metrics = PerformanceMetrics.from_trade_log([])
        assert metrics["win_rate_pct"] == 0.0

    def test_trade_metrics_all_wins(self):
        trades = [{"pnl": 500, "pnl_pct": 0.05} for _ in range(10)]
        metrics = PerformanceMetrics.from_trade_log(trades)
        assert metrics["win_rate_pct"] == 100.0
        assert metrics["total_pnl"] == 5000
        assert metrics["losing_trades"] == 0

    def test_profit_factor_no_losses(self):
        trades = [{"pnl": 100, "pnl_pct": 0.01} for _ in range(5)]
        metrics = PerformanceMetrics.from_trade_log(trades)
        assert metrics["profit_factor"] == float("inf")
