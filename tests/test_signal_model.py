"""
Unit tests for SignalModel — no GPU required.
Uses synthetic feature vectors to validate predict/explain interfaces.
"""
import numpy as np
import pandas as pd
import pytest

from features.feature_engineering import FEATURE_COLUMNS
from models.signal_model import SignalModel, LABEL_MAP


def make_feature_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(-1, 1, n) for col in FEATURE_COLUMNS}
    # Make some features more realistic
    data["rsi_14"] = rng.uniform(20, 80, n)
    data["volume_spike"] = rng.uniform(0.5, 3.0, n)
    data["avg_confidence"] = rng.uniform(0.4, 0.9, n)
    data["bullish_ratio"] = rng.uniform(0, 1, n)
    data["bearish_ratio"] = 1 - data["bullish_ratio"]
    return pd.DataFrame(data)


def make_labels(n: int = 500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(0, 3, n))


class TestSignalModel:
    def setup_method(self):
        self.X = make_feature_df()
        self.y = make_labels()
        self.model = SignalModel()
        self.model.fit(self.X, self.y)

    def test_predict_returns_list(self):
        preds = self.model.predict(self.X[:10])
        assert isinstance(preds, list)
        assert len(preds) == 10

    def test_predict_signal_values(self):
        preds = self.model.predict(self.X[:20])
        for p in preds:
            assert p["signal"] in ("BUY", "HOLD", "SELL")

    def test_confidence_in_range(self):
        preds = self.model.predict(self.X[:20])
        for p in preds:
            assert 0.0 <= p["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self):
        preds = self.model.predict(self.X[:10])
        for p in preds:
            total = sum(p["probabilities"].values())
            assert abs(total - 1.0) < 1e-4, f"Probs sum {total} != 1.0"

    def test_predict_single(self):
        vec = self.X.iloc[0]
        result = self.model.predict_single(vec)
        assert "signal" in result
        assert "confidence" in result

    def test_explain_returns_factors(self):
        explanations = self.model.explain(self.X[:5])
        assert len(explanations) == 5
        for exp in explanations:
            assert "top_factors" in exp
            assert len(exp["top_factors"]) <= 5

    def test_feature_importance(self):
        fi = self.model.feature_importance()
        assert list(fi.columns) == ["feature", "importance"]
        assert len(fi) == len(FEATURE_COLUMNS)
        assert fi["importance"].sum() > 0

    def test_save_load(self, tmp_path):
        path = tmp_path / "test_model.pkl"
        self.model.save(str(path))
        loaded = SignalModel()
        loaded.load(str(path))
        preds_orig = self.model.predict(self.X[:5])
        preds_loaded = loaded.predict(self.X[:5])
        for o, l in zip(preds_orig, preds_loaded):
            assert o["signal"] == l["signal"]
