"""
Technical indicator calculation from OHLCV price data.
Uses the `ta` library for standard indicators.
"""
import numpy as np
import pandas as pd
import ta
import structlog

logger = structlog.get_logger(__name__)


class TechnicalIndicators:
    """
    Computes a feature-rich set of technical indicators on a price DataFrame.
    Input DataFrame must have columns: open, high, low, close, volume
    with a DatetimeIndex.
    """

    # ── RSI ─────────────────────────────────────────────────────────────────
    @staticmethod
    def rsi(close: pd.Series, window: int = 14) -> pd.Series:
        return ta.momentum.RSIIndicator(close=close, window=window).rsi()

    # ── MACD ────────────────────────────────────────────────────────────────
    @staticmethod
    def macd(close: pd.Series) -> pd.DataFrame:
        macd_obj = ta.trend.MACD(close=close)
        return pd.DataFrame(
            {
                "macd": macd_obj.macd(),
                "macd_signal": macd_obj.macd_signal(),
                "macd_diff": macd_obj.macd_diff(),
            }
        )

    # ── Bollinger Bands ────────────────────────────────────────────────────
    @staticmethod
    def bollinger_bands(close: pd.Series, window: int = 20) -> pd.DataFrame:
        bb = ta.volatility.BollingerBands(close=close, window=window)
        return pd.DataFrame(
            {
                "bb_upper": bb.bollinger_hband(),
                "bb_middle": bb.bollinger_mavg(),
                "bb_lower": bb.bollinger_lband(),
                "bb_pct": bb.bollinger_pband(),
                "bb_width": bb.bollinger_wband(),
            }
        )

    # ── Volume indicators ──────────────────────────────────────────────────
    @staticmethod
    def volume_indicators(
        close: pd.Series, volume: pd.Series
    ) -> pd.DataFrame:
        obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        vwap = (close * volume).cumsum() / volume.cumsum()
        avg_volume_20 = volume.rolling(20).mean()
        volume_spike = volume / avg_volume_20
        return pd.DataFrame(
            {
                "obv": obv,
                "vwap": vwap,
                "volume_spike": volume_spike,
                "avg_volume_20": avg_volume_20,
            }
        )

    # ── Momentum ───────────────────────────────────────────────────────────
    @staticmethod
    def momentum(close: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "mom_1d": close.pct_change(1),
                "mom_5d": close.pct_change(5),
                "mom_10d": close.pct_change(10),
                "mom_20d": close.pct_change(20),
                "ema_9": ta.trend.EMAIndicator(close=close, window=9).ema_indicator(),
                "ema_21": ta.trend.EMAIndicator(close=close, window=21).ema_indicator(),
                "ema_50": ta.trend.EMAIndicator(close=close, window=50).ema_indicator(),
                "ema_200": ta.trend.EMAIndicator(close=close, window=200).ema_indicator(),
            }
        )

    # ── Volatility ─────────────────────────────────────────────────────────
    @staticmethod
    def volatility(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
        atr = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()
        returns = close.pct_change()
        realised_vol_20 = returns.rolling(20).std() * np.sqrt(252)
        realised_vol_5 = returns.rolling(5).std() * np.sqrt(252)
        return pd.DataFrame(
            {
                "atr": atr,
                "realised_vol_20": realised_vol_20,
                "realised_vol_5": realised_vol_5,
            }
        )

    # ── Trend ──────────────────────────────────────────────────────────────
    @staticmethod
    def trend_strength(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
        adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
        return pd.DataFrame(
            {
                "adx": adx.adx(),
                "adx_pos": adx.adx_pos(),
                "adx_neg": adx.adx_neg(),
            }
        )

    # ── Full feature set ───────────────────────────────────────────────────
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators on a price DataFrame.
        Expects columns: open, high, low, close, volume.
        Returns original df extended with indicator columns.
        """
        result = df.copy()

        result["rsi_14"] = self.rsi(df["close"])
        result = result.join(self.macd(df["close"]))
        result = result.join(self.bollinger_bands(df["close"]))
        result = result.join(self.volume_indicators(df["close"], df["volume"]))
        result = result.join(self.momentum(df["close"]))
        result = result.join(self.volatility(df["close"], df["high"], df["low"]))
        result = result.join(self.trend_strength(df["high"], df["low"], df["close"]))

        # Price position relative to MA
        result["price_vs_ema50"] = (df["close"] - result["ema_50"]) / result["ema_50"]
        result["price_vs_ema200"] = (df["close"] - result["ema_200"]) / result["ema_200"]

        logger.debug("technical_indicators_computed", rows=len(result), cols=len(result.columns))
        return result
