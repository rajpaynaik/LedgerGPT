from __future__ import annotations

"""
Trading signal model — XGBoost with SHAP explainability.
Outputs BUY / SELL / HOLD with confidence scores.
"""
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import structlog
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from config import get_settings
from features.feature_engineering import FEATURE_COLUMNS

logger = structlog.get_logger(__name__)

LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}
REVERSE_LABEL_MAP = {"SELL": 0, "HOLD": 1, "BUY": 2}

DEFAULT_XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "objective": "multi:softprob",
    "num_class": 3,
    "random_state": 42,
    "n_jobs": -1,
}


class SignalModel:
    """
    XGBoost multi-class classifier for BUY / HOLD / SELL signals.
    Wraps calibration and SHAP explanations.
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or DEFAULT_XGB_PARAMS
        self.model = xgb.XGBClassifier(**self.params)
        self.scaler = StandardScaler()
        self.calibrated: CalibratedClassifierCV | None = None
        self._explainer: shap.TreeExplainer | None = None
        self._is_fitted = False
        self.settings = get_settings()

    # ── Training ───────────────────────────────────────────────────────────
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "SignalModel":
        X_scaled = self.scaler.fit_transform(X_train[FEATURE_COLUMNS].fillna(0))

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val[FEATURE_COLUMNS].fillna(0))
            eval_set = [(X_val_scaled, y_val)]

        self.model.fit(
            X_scaled,
            y_train,
            eval_set=eval_set,
            verbose=50,
        )

        # Probability calibration
        self.calibrated = CalibratedClassifierCV(
            self.model, method="isotonic", cv="prefit"
        )
        self.calibrated.fit(X_scaled, y_train)

        # SHAP explainer
        self._explainer = shap.TreeExplainer(self.model)
        self._is_fitted = True
        logger.info("signal_model_trained", features=len(FEATURE_COLUMNS))
        return self

    # ── Inference ─────────────────────────────────────────────────────────
    def predict(self, X: pd.DataFrame | pd.Series) -> list[dict]:
        """
        Returns a list of signal dicts with confidence and explanation.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

        if isinstance(X, pd.Series):
            X = X.to_frame().T

        X_scaled = self.scaler.transform(X[FEATURE_COLUMNS].fillna(0))

        # Calibrated probabilities
        probs = self.calibrated.predict_proba(X_scaled)
        pred_classes = np.argmax(probs, axis=1)

        results = []
        for i, (cls, prob_row) in enumerate(zip(pred_classes, probs)):
            signal = LABEL_MAP[int(cls)]
            confidence = float(prob_row[cls])
            result = {
                "signal": signal,
                "confidence": round(confidence, 4),
                "probabilities": {
                    "SELL": round(float(prob_row[0]), 4),
                    "HOLD": round(float(prob_row[1]), 4),
                    "BUY": round(float(prob_row[2]), 4),
                },
            }
            results.append(result)
        return results

    def predict_single(self, feature_vector: pd.Series) -> dict:
        return self.predict(feature_vector.to_frame().T)[0]

    # ── SHAP Explanation ───────────────────────────────────────────────────
    def explain(self, X: pd.DataFrame, top_n: int = 5) -> list[dict]:
        """Return top SHAP feature importances per row."""
        if not self._explainer:
            raise RuntimeError("No explainer available. Fit the model first.")

        X_scaled = self.scaler.transform(X[FEATURE_COLUMNS].fillna(0))
        shap_values = self._explainer.shap_values(X_scaled)

        explanations = []
        # shap_values shape: (n_classes, n_samples, n_features)
        pred_classes = np.argmax(self.model.predict_proba(X_scaled), axis=1)

        for i, cls in enumerate(pred_classes):
            sv = shap_values[cls][i]
            feat_names = FEATURE_COLUMNS
            top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
            factors = [
                {
                    "feature": feat_names[j],
                    "shap_value": round(float(sv[j]), 4),
                    "direction": "positive" if sv[j] > 0 else "negative",
                }
                for j in top_idx
            ]
            explanations.append({"signal": LABEL_MAP[cls], "top_factors": factors})
        return explanations

    # ── Persistence ────────────────────────────────────────────────────────
    def save(self, path: str | None = None) -> Path:
        path = Path(path or self.settings.model_artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "calibrated": self.calibrated,
                    "scaler": self.scaler,
                    "params": self.params,
                },
                f,
            )
        logger.info("signal_model_saved", path=str(path))
        return path

    def load(self, path: str | None = None) -> "SignalModel":
        path = Path(path or self.settings.model_artifact_path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.calibrated = data["calibrated"]
        self.scaler = data["scaler"]
        self.params = data["params"]
        self._explainer = shap.TreeExplainer(self.model)
        self._is_fitted = True
        logger.info("signal_model_loaded", path=str(path))
        return self

    def feature_importance(self) -> pd.DataFrame:
        importance = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importance})
            .sort_values("importance", ascending=False)
        )
