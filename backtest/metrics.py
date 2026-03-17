"""
Performance metrics for backtesting results.
Calculates Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, etc.
"""
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.045  # 4.5% annualised


class PerformanceMetrics:
    """Compute and report strategy performance metrics from equity curve."""

    @staticmethod
    def from_equity_curve(equity: pd.Series) -> dict:
        """
        Compute all metrics from an equity curve (index=date, values=portfolio value).
        """
        returns = equity.pct_change().dropna()

        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        ann_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Sharpe
        excess = returns - (RISK_FREE_RATE / TRADING_DAYS_PER_YEAR)
        sharpe = (excess.mean() / returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if returns.std() != 0 else 0.0

        # Sortino
        downside = returns[returns < 0].std()
        sortino = (excess.mean() / downside) * np.sqrt(TRADING_DAYS_PER_YEAR) if downside != 0 else 0.0

        # Max drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        max_drawdown_duration = PerformanceMetrics._drawdown_duration(equity)

        # Calmar
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # VaR / CVaR (95%)
        var_95 = float(np.percentile(returns, 5))
        cvar_95 = float(returns[returns <= var_95].mean())

        return {
            "total_return_pct": round(total_return * 100, 2),
            "annualised_return_pct": round(ann_return * 100, 2),
            "annualised_volatility_pct": round(ann_vol * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "max_drawdown_duration_days": max_drawdown_duration,
            "var_95_pct": round(var_95 * 100, 2),
            "cvar_95_pct": round(cvar_95 * 100, 2),
            "start": str(equity.index[0]),
            "end": str(equity.index[-1]),
        }

    @staticmethod
    def from_trade_log(trades: list[dict]) -> dict:
        """Trade-level statistics: win rate, avg win/loss, profit factor."""
        if not trades:
            return {"win_rate_pct": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0, "profit_factor": 0.0}

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) * 100
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        profit_factor = abs(sum(wins) / sum(losses)) if losses else float("inf")

        return {
            "total_trades": len(pnls),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 3),
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(np.mean(pnls), 2),
        }

    @staticmethod
    def _drawdown_duration(equity: pd.Series) -> int:
        """Return length (in days) of the longest drawdown period."""
        rolling_max = equity.cummax()
        in_dd = equity < rolling_max
        if not in_dd.any():
            return 0
        groups = (in_dd != in_dd.shift()).cumsum()
        dd_lengths = in_dd.groupby(groups).sum()
        return int(dd_lengths.max())

    @staticmethod
    def print_report(equity_metrics: dict, trade_metrics: dict) -> None:
        print("\n" + "=" * 50)
        print("  LedgerGPT Backtest Report")
        print("=" * 50)
        for k, v in equity_metrics.items():
            print(f"  {k:<35} {v}")
        print("-" * 50)
        for k, v in trade_metrics.items():
            print(f"  {k:<35} {v}")
        print("=" * 50 + "\n")
