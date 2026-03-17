"""
Rule-based weighted scoring model.
Used as:
  1. A baseline to compare against the ML model.
  2. A fallback when the ML model artifact is unavailable.
  3. An explainable sanity check on signals.

Weights match the original LedgerGPT spec:
  Momentum     25%
  RSI          20%
  MA Crossover 20%
  Volume Spike 15%
  Sentiment    12%
  Volatility    8%
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Signal thresholds
BUY_THRESHOLD  = 0.55   # aggregate score ≥ 0.55 → BUY
SELL_THRESHOLD = 0.45   # aggregate score ≤ 0.45 → SELL


@dataclass
class WeightedScorerConfig:
    momentum_weight:     float = 0.25
    rsi_weight:          float = 0.20
    ma_crossover_weight: float = 0.20
    volume_spike_weight: float = 0.15
    sentiment_weight:    float = 0.12
    volatility_weight:   float = 0.08

    buy_threshold:  float = BUY_THRESHOLD
    sell_threshold: float = SELL_THRESHOLD

    # RSI thresholds
    rsi_oversold:   float = 30.0
    rsi_overbought: float = 70.0
    # Volume spike threshold
    volume_spike_threshold: float = 1.5
    # Volatility penalty threshold
    high_volatility_threshold: float = 0.30


class WeightedScorer:
    """
    Deterministic rule-based scorer.
    Each component returns a sub-score in [0, 1]:
      0.0 = strongly bearish
      0.5 = neutral
      1.0 = strongly bullish
    The final score is a weighted average → mapped to BUY/HOLD/SELL.
    """

    def __init__(self, config: WeightedScorerConfig | None = None) -> None:
        self.cfg = config or WeightedScorerConfig()

    # ── Sub-scorers ────────────────────────────────────────────────────────
    def _momentum_score(self, row: pd.Series) -> float:
        mom = row.get("mom_5d", 0.0)
        if pd.isna(mom):
            return 0.5
        # Sigmoid-like mapping: ±10% range maps to 0..1
        return float(np.clip(0.5 + mom * 5.0, 0.0, 1.0))

    def _rsi_score(self, row: pd.Series) -> float:
        rsi = row.get("rsi_14", 50.0)
        if pd.isna(rsi):
            return 0.5
        if rsi <= self.cfg.rsi_oversold:
            return 1.0   # oversold → bullish entry signal
        if rsi >= self.cfg.rsi_overbought:
            return 0.0   # overbought → bearish
        # Linear interpolation in [oversold, overbought]
        return 1.0 - (rsi - self.cfg.rsi_oversold) / (
            self.cfg.rsi_overbought - self.cfg.rsi_oversold
        )

    def _ma_crossover_score(self, row: pd.Series) -> float:
        """Score based on price position relative to EMA50 and EMA200."""
        p50  = row.get("price_vs_ema50", 0.0)
        p200 = row.get("price_vs_ema200", 0.0)
        if pd.isna(p50) or pd.isna(p200):
            return 0.5
        # Both above MAs → bullish; both below → bearish
        score = 0.5 + np.clip(p50 * 2.0, -0.3, 0.3) + np.clip(p200 * 1.5, -0.2, 0.2)
        return float(np.clip(score, 0.0, 1.0))

    def _volume_spike_score(self, row: pd.Series) -> float:
        spike = row.get("volume_spike", 1.0)
        if pd.isna(spike):
            return 0.5
        # High volume with positive price momentum = bullish confirmation
        mom = row.get("mom_1d", 0.0)
        if spike >= self.cfg.volume_spike_threshold and mom > 0:
            return min(0.5 + spike * 0.1, 1.0)
        if spike >= self.cfg.volume_spike_threshold and mom < 0:
            return max(0.5 - spike * 0.1, 0.0)
        return 0.5

    def _sentiment_score(self, row: pd.Series) -> float:
        overall = row.get("overall_sentiment", 0.0)
        if pd.isna(overall):
            return 0.5
        # overall_sentiment is in [-1, 1] → map to [0, 1]
        return float(np.clip(0.5 + overall * 0.5, 0.0, 1.0))

    def _volatility_score(self, row: pd.Series) -> float:
        """High volatility penalises the score (uncertainty)."""
        vol = row.get("realised_vol_20", 0.15)
        if pd.isna(vol):
            return 0.5
        if vol >= self.cfg.high_volatility_threshold:
            return 0.3   # very volatile → lean toward HOLD
        # Below threshold: neutral score (volatility alone doesn't predict direction)
        return 0.5

    # ── Composite score ────────────────────────────────────────────────────
    def score_row(self, row: pd.Series) -> dict:
        cfg = self.cfg
        sub_scores = {
            "momentum":     self._momentum_score(row),
            "rsi":          self._rsi_score(row),
            "ma_crossover": self._ma_crossover_score(row),
            "volume_spike": self._volume_spike_score(row),
            "sentiment":    self._sentiment_score(row),
            "volatility":   self._volatility_score(row),
        }
        weights = {
            "momentum":     cfg.momentum_weight,
            "rsi":          cfg.rsi_weight,
            "ma_crossover": cfg.ma_crossover_weight,
            "volume_spike": cfg.volume_spike_weight,
            "sentiment":    cfg.sentiment_weight,
            "volatility":   cfg.volatility_weight,
        }
        composite = sum(sub_scores[k] * weights[k] for k in sub_scores)

        if composite >= cfg.buy_threshold:
            signal = "BUY"
        elif composite <= cfg.sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Confidence: distance from nearest threshold, mapped to [0.5, 1.0]
        dist_buy  = abs(composite - cfg.buy_threshold)
        dist_sell = abs(composite - cfg.sell_threshold)
        dist = min(dist_buy, dist_sell) if signal == "HOLD" else max(dist_buy, dist_sell)
        confidence = float(np.clip(0.5 + dist * 2.0, 0.5, 0.95))

        return {
            "signal":      signal,
            "confidence":  round(confidence, 3),
            "composite":   round(float(composite), 4),
            "sub_scores":  {k: round(v, 3) for k, v in sub_scores.items()},
            "model":       "weighted_scorer",
        }

    def predict(self, X: pd.DataFrame) -> list[dict]:
        return [self.score_row(X.iloc[i]) for i in range(len(X))]

    def predict_single(self, feature_vector: pd.Series) -> dict:
        return self.score_row(feature_vector)

    def compare_with_ml(
        self,
        X: pd.DataFrame,
        ml_signals: list[dict],
    ) -> pd.DataFrame:
        """
        Side-by-side comparison of weighted scorer vs ML model.
        Useful for sanity checking and regime-specific diagnostics.
        """
        rule_signals = self.predict(X)
        rows = []
        for i, (rule, ml) in enumerate(zip(rule_signals, ml_signals)):
            rows.append({
                "rule_signal":    rule["signal"],
                "rule_conf":      rule["confidence"],
                "ml_signal":      ml["signal"],
                "ml_conf":        ml["confidence"],
                "agree":          rule["signal"] == ml["signal"],
                "rule_composite": rule.get("composite", 0.0),
            })
        return pd.DataFrame(rows)
