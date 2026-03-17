"""
Unit tests for TechnicalIndicators.
Uses synthetic OHLCV data — no market data API calls.
"""
import numpy as np
import pandas as pd
import pytest

from features.technical_indicators import TechnicalIndicators


def make_price_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1)  # ensure positive
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.uniform(-0.01, 0.01, n))
    volume = rng.integers(100_000, 10_000_000, n).astype(float)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class TestTechnicalIndicators:
    def setup_method(self):
        self.ti = TechnicalIndicators()
        self.df = make_price_df()

    def test_rsi_range(self):
        rsi = self.ti.rsi(self.df["close"])
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd_columns(self):
        result = self.ti.macd(self.df["close"])
        assert set(result.columns) == {"macd", "macd_signal", "macd_diff"}
        assert len(result) == len(self.df)

    def test_bollinger_bands_ordering(self):
        bb = self.ti.bollinger_bands(self.df["close"])
        valid = bb.dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_volume_spike_positive(self):
        result = self.ti.volume_indicators(self.df["close"], self.df["volume"])
        assert (result["volume_spike"].dropna() > 0).all()

    def test_compute_all_shape(self):
        result = self.ti.compute_all(self.df)
        assert len(result) == len(self.df)
        # Check key columns present
        for col in ["rsi_14", "macd", "bb_pct", "volume_spike", "mom_1d", "adx"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_momentum_returns_correct_length(self):
        result = self.ti.momentum(self.df["close"])
        assert len(result) == len(self.df)

    def test_no_inf_values(self):
        result = self.ti.compute_all(self.df)
        numeric = result.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Infinite values found in indicators"
