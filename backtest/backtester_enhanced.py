from __future__ import annotations

"""
Backtesting framework for signal validation.
Tests trading strategy on historical data with sentiment context.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import json
import numpy as np
import pandas as pd
import yfinance as yf
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    entry_price: float
    entry_date: datetime
    exit_price: float
    exit_date: datetime
    signal_type: str  # BUY, SELL
    returns_pct: float
    holding_days: int


@dataclass
class BacktestResults:
    """Backtest performance summary."""
    ticker: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    total_returns_pct: float
    benchmark_returns_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    avg_trade_return_pct: float
    profit_factor: float
    mdd: float
    
    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': f"{self.win_rate_pct:.1f}%",
            'total_returns': f"{self.total_returns_pct:.1f}%",
            'benchmark_returns': f"{self.benchmark_returns_pct:.1f}%",
            'sharpe_ratio': round(self.sharpe_ratio, 3),
            'max_drawdown': f"{self.max_drawdown_pct:.1f}%",
            'avg_return_per_trade': f"{self.avg_trade_return_pct:.1f}%",
            'profit_factor': round(self.profit_factor, 2)
        }


class Backtester:
    """
    Backtesting engine for signal-based trading strategy.
    Evaluates historical performance of signal predictions.
    """
    
    def __init__(self, initial_capital: float = 10000):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital for simulation
        """
        self.initial_capital = initial_capital
        self.trades: list[Trade] = []
    
    def backtest(
        self,
        ticker: str,
        period: str = "1y",
        signal_func=None  # Function that generates signals for a price dataframe
    ) -> BacktestResults:
        """
        Run backtest on historical data.
        
        Args:
            ticker: Stock ticker
            period: Historical period ('1y', '2y', etc.)
            signal_func: Function(df) -> df with 'signal' column added
            
        Returns:
            BacktestResults summary
        """
        try:
            # Fetch historical data
            df = yf.download(ticker, period=period, progress=False)
            
            if len(df) < 50:
                logger.error("Insufficient data", ticker=ticker, rows=len(df))
                return self._empty_results(ticker)
            
            # Generate signals (if function provided)
            if signal_func:
                df = signal_func(df)
            else:
                # Simple signal: +2% change = BUY, -2% = SELL, else HOLD
                df['returns'] = df['Close'].pct_change()
                df['signal'] = 0  # HOLD
                df.loc[df['returns'] > 0.02, 'signal'] = 1   # BUY
                df.loc[df['returns'] < -0.02, 'signal'] = -1  # SELL
                df['signal'] = df['signal'].shift(1)  # Look-ahead
            
            # Run simulation
            self._simulate_trades(df)
            
            # Calculate metrics
            results = self._calculate_metrics(df, ticker)
            
            logger.info("Backtest completed", ticker=ticker, results=results.to_dict())
            return results
        
        except Exception as e:
            logger.error("Backtest failed", error=str(e), ticker=ticker)
            return self._empty_results(ticker)
    
    def _simulate_trades(self, df: pd.DataFrame):
        """
        Simulate trades based on signals.
        
        Args:
            df: DataFrame with 'signal' column
        """
        self.trades = []
        
        if 'signal' not in df.columns:
            return
        
        position = None
        entry_price = None
        entry_date = None
        
        for idx, row in df.iterrows():
            signal = row.get('signal', 0)
            price = row['Close']
            date = idx
            
            # BUY signal
            if signal == 1 and position is None:
                position = 'LONG'
                entry_price = price
                entry_date = date
            
            # SELL signal or stop loss
            elif signal == -1 and position == 'LONG':
                exit_price = price
                returns_pct = (exit_price - entry_price) / entry_price * 100
                holding_days = (date - entry_date).days
                
                self.trades.append(Trade(
                    entry_price=entry_price,
                    entry_date=entry_date,
                    exit_price=exit_price,
                    exit_date=date,
                    signal_type='BUY',
                    returns_pct=returns_pct,
                    holding_days=max(1, holding_days)
                ))
                
                position = None
                entry_price = None
                entry_date = None
        
        # Close position at end
        if position == 'LONG':
            final_price = df['Close'].iloc[-1]
            final_date = df.index[-1]
            returns_pct = (final_price - entry_price) / entry_price * 100
            holding_days = (final_date - entry_date).days
            
            self.trades.append(Trade(
                entry_price=entry_price,
                entry_date=entry_date,
                exit_price=final_price,
                exit_date=final_date,
                signal_type='BUY',
                returns_pct=returns_pct,
                holding_days=max(1, holding_days)
            ))
    
    def _calculate_metrics(self, df: pd.DataFrame, ticker: str) -> BacktestResults:
        """
        Calculate backtest performance metrics.
        
        Args:
            df: Price dataframe
            ticker: Stock ticker
            
        Returns:
            BacktestResults
        """
        if not self.trades:
            return self._empty_results(ticker)
        
        # Trade statistics
        returns = [t.returns_pct for t in self.trades]
        winning = [r for r in returns if r > 0]
        losing = [r for r in returns if r <= 0]
        
        total_profit = sum(winning)
        total_loss = abs(sum(losing))
        
        win_rate = len(winning) / len(self.trades) * 100 if self.trades else 0
        avg_return = np.mean(returns) if returns else 0
        sharpe = self._calc_sharpe_ratio(returns)
        mdd = self._calc_max_drawdown(df)
        
        # Benchmark: buy and hold
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        bench_returns = (end_price - start_price) / start_price * 100
        
        # Strategy returns (simplified: cumulative trade returns)
        strat_returns = sum(returns)
        
        # Profit factor
        profit_factor = total_profit / total_loss if total_loss > 0 else (1.0 if total_profit > 0 else 0)
        
        return BacktestResults(
            ticker=ticker,
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate_pct=win_rate,
            total_returns_pct=strat_returns,
            benchmark_returns_pct=bench_returns,
            sharpe_ratio=sharpe,
            max_drawdown_pct=mdd,
            avg_trade_return_pct=avg_return,
            profit_factor=profit_factor,
            mdd=mdd
        )
    
    def _calc_sharpe_ratio(self, returns: list[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0
        
        excess_returns = np.array(returns) / 100 - (risk_free_rate / 252)
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)) if np.std(excess_returns) > 0 else 0
    
    def _calc_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown percentage."""
        if len(df) < 2:
            return 0
        
        prices = df['Close'].values
        running_max = np.maximum.accumulate(prices)
        drawdown = (prices - running_max) / running_max
        max_dd = np.min(drawdown) * 100
        return max_dd
    
    def _empty_results(self, ticker: str) -> BacktestResults:
        """Return empty backtest results."""
        return BacktestResults(
            ticker=ticker,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0,
            total_returns_pct=0,
            benchmark_returns_pct=0,
            sharpe_ratio=0,
            max_drawdown_pct=0,
            avg_trade_return_pct=0,
            profit_factor=0,
            mdd=0
        )
    
    def get_trades_summary(self) -> list[dict]:
        """Get summary of all trades."""
        return [
            {
                'entry_date': str(t.entry_date)[:10],
                'exit_date': str(t.exit_date)[:10],
                'entry_price': round(t.entry_price, 2),
                'exit_price': round(t.exit_price, 2),
                'return_pct': round(t.returns_pct, 2),
                'holding_days': t.holding_days
            }
            for t in self.trades
        ]
