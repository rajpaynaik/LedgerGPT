from __future__ import annotations

"""
Feature Engineering pipeline.
Merges technical indicators + sentiment features into a unified feature matrix
ready for the ML decision model.
"""
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from config import get_settings
from .technical_indicators import TechnicalIndicators
from .sentiment_aggregator import SentimentAggregator

logger = structlog.get_logger(__name__)

# Ordered list of feature columns fed to the ML model
FEATURE_COLUMNS = [
    # Sentiment
    "twitter_sentiment",
    "reddit_sentiment",
    "news_sentiment",
    "overall_sentiment",
    "bullish_ratio",
    "bearish_ratio",
    "sentiment_momentum",
    "avg_confidence",
    "tweet_count",
    "reddit_count",
    "news_count",
    "high_urgency_count",
    # Technical
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "bb_pct",
    "bb_width",
    "volume_spike",
    "obv",
    "mom_1d",
    "mom_5d",
    "mom_10d",
    "mom_20d",
    "ema_9",
    "ema_21",
    "realised_vol_20",
    "realised_vol_5",
    "atr",
    "adx",
    "price_vs_ema50",
    "price_vs_ema200",
]

# Forward-return window for label generation (trading days)
LABEL_HORIZON_DAYS = 5


class FeatureEngineer:
    """
    Builds the feature matrix used to train and score the ML signal model.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.tech = TechnicalIndicators()
        self.sent_agg = SentimentAggregator()

    # ── Label generation ───────────────────────────────────────────────────
    @staticmethod
    def generate_labels(
        price_df: pd.DataFrame,
        horizon: int = LABEL_HORIZON_DAYS,
        buy_threshold: float = 0.02,
        sell_threshold: float = -0.02,
    ) -> pd.Series:
        """
        Generate BUY / SELL / HOLD labels from forward returns.
        Labels: 2 = BUY, 1 = HOLD, 0 = SELL
        """
        fwd_return = price_df["close"].pct_change(horizon).shift(-horizon)
        labels = pd.Series(1, index=price_df.index, name="label", dtype=int)
        labels[fwd_return >= buy_threshold] = 2
        labels[fwd_return <= sell_threshold] = 0
        return labels

    # ── Build training dataset ─────────────────────────────────────────────
    def build_training_dataset(
        self,
        price_df: pd.DataFrame,
        sentiment_records: list[dict],
        ticker: str,
    ) -> pd.DataFrame:
        """
        Full pipeline: price OHLCV + sentiment records → feature matrix with labels.
        """
        # Technical indicators
        tech_df = self.tech.compute_all(price_df)

        # Sentiment features (one row per trading day)
        sent_rows = []
        for ts in price_df.index:
            as_of = ts.to_pydatetime().replace(tzinfo=timezone.utc)
            agg = self.sent_agg.aggregate(sentiment_records, ticker, as_of=as_of)
            agg["timestamp"] = ts
            sent_rows.append(agg)

        sent_df = pd.DataFrame(sent_rows).set_index("timestamp")
        sent_df = sent_df.drop(columns=["ticker", "as_of"], errors="ignore")

        # Merge
        merged = tech_df.join(sent_df, how="left")

        # Labels
        merged["label"] = self.generate_labels(price_df)

        # Drop NaN rows (from indicator warm-up and label horizon)
        merged = merged.dropna(subset=["rsi_14", "macd", "label"])

        logger.info(
            "training_dataset_built",
            ticker=ticker,
            rows=len(merged),
            features=len(FEATURE_COLUMNS),
        )
        return merged

    # ── Build inference vector ─────────────────────────────────────────────
    def build_inference_vector(
        self,
        price_df: pd.DataFrame,
        sentiment_records: list[dict],
        ticker: str,
    ) -> pd.Series:
        """
        Build a single feature vector for the most recent bar.
        Used at inference time for live signal generation.
        """
        tech_df = self.tech.compute_all(price_df)
        latest_tech = tech_df.iloc[-1]

        sent_agg = self.sent_agg.aggregate(sentiment_records, ticker)
        sent_series = pd.Series(sent_agg).drop(["ticker", "as_of"], errors="ignore")

        combined = pd.concat([latest_tech, sent_series])
        vector = combined[FEATURE_COLUMNS]
        vector = vector.fillna(0.0)
        
        # Ensure we return a proper 1D Series (not DataFrame)
        if isinstance(vector, pd.DataFrame):
            vector = vector.iloc[:, 0]
        return pd.Series(vector)

    # ── Persist / load ─────────────────────────────────────────────────────
    def save_features(self, df: pd.DataFrame, ticker: str) -> Path:
        out_dir = Path(self.settings.feature_store_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{ticker}_features.parquet"
        df.to_parquet(path)
        logger.info("features_saved", path=str(path), rows=len(df))
        return path

    def load_features(self, ticker: str) -> pd.DataFrame:
        path = Path(self.settings.feature_store_path) / f"{ticker}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        df = pd.read_parquet(path)
        logger.info("features_loaded", path=str(path), rows=len(df))
        return df
