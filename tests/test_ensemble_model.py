"""
Unit tests for EnsembleSignalModel.
Uses synthetic data — no GPU required.
"""
import numpy as np
import pandas as pd
import pytest

from features.feature_engineering import FEATURE_COLUMNS
from models.ensemble_model import EnsembleSignalModel


def make_X(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(-1, 1, n) for col in FEATURE_COLUMNS}
    data["rsi_14"] = rng.uniform(20, 80, n)
    data["volume_spike"] = rng.uniform(0.5, 3.0, n)
    data["bullish_ratio"] = rng.uniform(0, 1, n)
    data["earnings_event"] = rng.integers(0, 2, n).astype(float)
    data["ma_event"] = rng.integers(0, 2, n).astype(float)
    data["regulatory_event"] = rng.integers(0, 2, n).astype(float)
    data["sentiment_velocity"] = rng.uniform(-0.5, 0.5, n)
    data["sentiment_volume"] = rng.integers(0, 500, n).astype(float)
    return pd.DataFrame(data)


def make_y(n: int = 300, seed: int = 1) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).integers(0, 3, n))


class TestEnsembleSignalModel:
    def setup_method(self):
        self.X = make_X()
        self.y = make_y()
        self.model = EnsembleSignalModel()
        self.model.fit(self.X, self.y)

    def test_predict_length(self):
        preds = self.model.predict(self.X[:10])
        assert len(preds) == 10

    def test_predict_signal_values(self):
        preds = self.model.predict(self.X[:20])
        for p in preds:
            assert p["signal"] in ("BUY", "HOLD", "SELL")
            assert p["model"] == "ensemble"

    def test_probabilities_sum_to_one(self):
        preds = self.model.predict(self.X[:10])
        for p in preds:
            total = sum(p["probabilities"].values())
            assert abs(total - 1.0) < 1e-3

    def test_individual_signals(self):
        row = self.X.iloc[0]
        result = self.model.individual_signals(row)
        assert set(result.keys()) == {"xgboost", "lightgbm", "random_forest"}
        for sig in result.values():
            assert sig in ("BUY", "HOLD", "SELL")

    def test_explain_top_factors(self):
        exps = self.model.explain(self.X[:5])
        assert len(exps) == 5
        for exp in exps:
            assert "top_factors" in exp
            assert len(exp["top_factors"]) <= 5

    def test_feature_importance_covers_new_features(self):
        fi = self.model.feature_importance()
        feature_names = fi["feature"].tolist()
        for new_feat in ("earnings_event", "ma_event", "regulatory_event",
                         "sentiment_velocity", "sentiment_volume"):
            assert new_feat in feature_names, f"{new_feat} missing from feature importance"

    def test_save_load(self, tmp_path):
        path = tmp_path / "ensemble.pkl"
        self.model.save(str(path))
        loaded = EnsembleSignalModel()
        loaded.load(str(path))
        preds_orig   = self.model.predict(self.X[:5])
        preds_loaded = loaded.predict(self.X[:5])
        for o, l in zip(preds_orig, preds_loaded):
            assert o["signal"] == l["signal"]
