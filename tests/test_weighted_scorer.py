"""
Unit tests for WeightedScorer — no external dependencies.
"""
import pandas as pd
import pytest

from models.weighted_scorer import WeightedScorer, WeightedScorerConfig


def base_row(**kwargs) -> pd.Series:
    defaults = {
        "rsi_14": 50.0,
        "mom_1d": 0.0,
        "mom_5d": 0.0,
        "price_vs_ema50": 0.0,
        "price_vs_ema200": 0.0,
        "volume_spike": 1.0,
        "realised_vol_20": 0.15,
        "overall_sentiment": 0.0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


class TestWeightedScorer:
    def setup_method(self):
        self.scorer = WeightedScorer()

    def test_strong_buy_signal(self):
        row = base_row(
            rsi_14=25.0,         # oversold
            mom_5d=0.06,         # strong upward momentum
            price_vs_ema50=0.04, # above EMA50
            volume_spike=2.5,    # volume spike with positive momentum
            overall_sentiment=0.6,
        )
        result = self.scorer.score_row(row)
        assert result["signal"] == "BUY"
        assert result["confidence"] > 0.5

    def test_strong_sell_signal(self):
        row = base_row(
            rsi_14=78.0,          # overbought
            mom_5d=-0.07,         # strong downward momentum
            price_vs_ema50=-0.05, # below EMA50
            overall_sentiment=-0.7,
        )
        result = self.scorer.score_row(row)
        assert result["signal"] == "SELL"

    def test_neutral_is_hold(self):
        row = base_row()  # all defaults → neutral
        result = self.scorer.score_row(row)
        assert result["signal"] == "HOLD"

    def test_high_volatility_penalises(self):
        # Two identical rows except volatility
        row_normal = base_row(rsi_14=25.0, mom_5d=0.05, overall_sentiment=0.5)
        row_volatile = base_row(rsi_14=25.0, mom_5d=0.05, overall_sentiment=0.5, realised_vol_20=0.40)
        res_normal   = self.scorer.score_row(row_normal)
        res_volatile = self.scorer.score_row(row_volatile)
        assert res_normal["composite"] > res_volatile["composite"]

    def test_sub_scores_present(self):
        result = self.scorer.score_row(base_row())
        for key in ("momentum", "rsi", "ma_crossover", "volume_spike", "sentiment", "volatility"):
            assert key in result["sub_scores"]

    def test_predict_batch(self):
        df = pd.DataFrame([base_row().to_dict() for _ in range(10)])
        results = self.scorer.predict(df)
        assert len(results) == 10
        for r in results:
            assert r["signal"] in ("BUY", "HOLD", "SELL")
            assert 0.5 <= r["confidence"] <= 1.0

    def test_compare_with_ml(self):
        df = pd.DataFrame([base_row().to_dict() for _ in range(5)])
        ml_signals = [{"signal": "HOLD", "confidence": 0.6} for _ in range(5)]
        comparison = self.scorer.compare_with_ml(df, ml_signals)
        assert list(comparison.columns) == [
            "rule_signal", "rule_conf", "ml_signal", "ml_conf", "agree", "rule_composite"
        ]
        assert len(comparison) == 5
