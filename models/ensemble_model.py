"""
Ensemble signal model: XGBoost + LightGBM + Random Forest with soft voting.
Produces more robust signals than any single model, especially at distribution shift.
"""
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import structlog
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config import get_settings
from features.feature_engineering import FEATURE_COLUMNS

logger = structlog.get_logger(__name__)

LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}

# ── Per-model defaults ────────────────────────────────────────────────────────
DEFAULT_XGB = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
    "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "tree_method": "hist", "objective": "multi:softprob", "num_class": 3,
    "eval_metric": "mlogloss", "use_label_encoder": False,
    "random_state": 42, "n_jobs": -1,
}

DEFAULT_LGB = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
    "num_leaves": 63, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "objective": "multiclass", "num_class": 3,
    "metric": "multi_logloss", "random_state": 42, "n_jobs": -1,
    "verbose": -1,
}

DEFAULT_RF = {
    "n_estimators": 300, "max_depth": 10, "min_samples_split": 5,
    "min_samples_leaf": 3, "max_features": "sqrt",
    "random_state": 42, "n_jobs": -1,
}

# Weights for soft voting (XGBoost > LightGBM > RF based on typical finance perf)
ENSEMBLE_WEIGHTS = [0.45, 0.35, 0.20]


class EnsembleSignalModel:
    """
    Soft-voting ensemble of XGBoost, LightGBM, and Random Forest.
    Each model is independently calibrated, then their probabilities
    are combined via weighted average.
    """

    def __init__(
        self,
        xgb_params: dict | None = None,
        lgb_params: dict | None = None,
        rf_params: dict | None = None,
        weights: list[float] | None = None,
    ) -> None:
        self.settings = get_settings()
        self.weights = weights or ENSEMBLE_WEIGHTS
        self.scaler = StandardScaler()

        self._xgb = xgb.XGBClassifier(**(xgb_params or DEFAULT_XGB))
        self._lgb = lgb.LGBMClassifier(**(lgb_params or DEFAULT_LGB))
        self._rf = RandomForestClassifier(**(rf_params or DEFAULT_RF))

        self._cal_xgb: CalibratedClassifierCV | None = None
        self._cal_lgb: CalibratedClassifierCV | None = None
        self._cal_rf: CalibratedClassifierCV | None = None

        self._xgb_explainer: shap.TreeExplainer | None = None
        self._is_fitted = False

    # ── Training ───────────────────────────────────────────────────────────
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "EnsembleSignalModel":
        X_sc = self.scaler.fit_transform(X_train[FEATURE_COLUMNS].fillna(0))

        # XGBoost
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_sc = self.scaler.transform(X_val[FEATURE_COLUMNS].fillna(0))
            eval_set = [(X_val_sc, y_val)]

        self._xgb.fit(X_sc, y_train, eval_set=eval_set, verbose=False)
        self._cal_xgb = CalibratedClassifierCV(self._xgb, method="isotonic", cv="prefit")
        self._cal_xgb.fit(X_sc, y_train)

        # LightGBM
        if eval_set is not None:
            self._lgb.fit(
                X_sc, y_train,
                eval_set=[(X_val_sc, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
        else:
            self._lgb.fit(X_sc, y_train)
        self._cal_lgb = CalibratedClassifierCV(self._lgb, method="isotonic", cv="prefit")
        self._cal_lgb.fit(X_sc, y_train)

        # Random Forest (no early stopping)
        self._rf.fit(X_sc, y_train)
        self._cal_rf = CalibratedClassifierCV(self._rf, method="isotonic", cv="prefit")
        self._cal_rf.fit(X_sc, y_train)

        # SHAP explainer backed by XGBoost (fastest)
        self._xgb_explainer = shap.TreeExplainer(self._xgb)
        self._is_fitted = True
        logger.info("ensemble_model_trained", features=len(FEATURE_COLUMNS))
        return self

    # ── Inference ─────────────────────────────────────────────────────────
    def _get_probs(self, X_scaled: np.ndarray) -> np.ndarray:
        """Weighted average of calibrated probabilities from all three models."""
        p_xgb = self._cal_xgb.predict_proba(X_scaled)
        p_lgb = self._cal_lgb.predict_proba(X_scaled)
        p_rf  = self._cal_rf.predict_proba(X_scaled)
        w = self.weights
        return w[0] * p_xgb + w[1] * p_lgb + w[2] * p_rf

    def predict(self, X: pd.DataFrame | pd.Series) -> list[dict]:
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() or load() first.")
        if isinstance(X, pd.Series):
            X = X.to_frame().T

        X_sc = self.scaler.transform(X[FEATURE_COLUMNS].fillna(0))
        probs = self._get_probs(X_sc)
        pred_classes = np.argmax(probs, axis=1)

        results = []
        for cls, prob_row in zip(pred_classes, probs):
            signal = LABEL_MAP[int(cls)]
            results.append({
                "signal": signal,
                "confidence": round(float(prob_row[cls]), 4),
                "probabilities": {
                    "SELL": round(float(prob_row[0]), 4),
                    "HOLD": round(float(prob_row[1]), 4),
                    "BUY":  round(float(prob_row[2]), 4),
                },
                "model": "ensemble",
            })
        return results

    def predict_single(self, feature_vector: pd.Series) -> dict:
        return self.predict(feature_vector.to_frame().T)[0]

    # ── SHAP explanation (from XGBoost sub-model) ──────────────────────────
    def explain(self, X: pd.DataFrame, top_n: int = 5) -> list[dict]:
        if not self._xgb_explainer:
            raise RuntimeError("No explainer. Fit the model first.")
        X_sc = self.scaler.transform(X[FEATURE_COLUMNS].fillna(0))
        shap_values = self._xgb_explainer.shap_values(X_sc)
        pred_classes = np.argmax(self._xgb.predict_proba(X_sc), axis=1)

        explanations = []
        for i, cls in enumerate(pred_classes):
            sv = shap_values[cls][i]
            top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
            explanations.append({
                "signal": LABEL_MAP[cls],
                "model": "ensemble",
                "top_factors": [
                    {
                        "feature": FEATURE_COLUMNS[j],
                        "shap_value": round(float(sv[j]), 4),
                        "direction": "positive" if sv[j] > 0 else "negative",
                    }
                    for j in top_idx
                ],
            })
        return explanations

    # ── Individual model breakdown ─────────────────────────────────────────
    def individual_signals(self, X: pd.DataFrame | pd.Series) -> dict:
        """Return predictions from each sub-model for comparison / debugging."""
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        X_sc = self.scaler.transform(X[FEATURE_COLUMNS].fillna(0))
        return {
            "xgboost": LABEL_MAP[int(np.argmax(self._cal_xgb.predict_proba(X_sc), axis=1)[0])],
            "lightgbm": LABEL_MAP[int(np.argmax(self._cal_lgb.predict_proba(X_sc), axis=1)[0])],
            "random_forest": LABEL_MAP[int(np.argmax(self._cal_rf.predict_proba(X_sc), axis=1)[0])],
        }

    # ── Persistence ────────────────────────────────────────────────────────
    def save(self, path: str | None = None) -> Path:
        base = Path(self.settings.model_artifact_path).parent
        path = Path(path or (base / "ensemble_model.pkl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "xgb": self._xgb, "lgb": self._lgb, "rf": self._rf,
                "cal_xgb": self._cal_xgb, "cal_lgb": self._cal_lgb, "cal_rf": self._cal_rf,
                "scaler": self.scaler, "weights": self.weights,
            }, f)
        logger.info("ensemble_model_saved", path=str(path))
        return path

    def load(self, path: str | None = None) -> "EnsembleSignalModel":
        base = Path(self.settings.model_artifact_path).parent
        path = Path(path or (base / "ensemble_model.pkl"))
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._xgb = data["xgb"]; self._lgb = data["lgb"]; self._rf = data["rf"]
        self._cal_xgb = data["cal_xgb"]; self._cal_lgb = data["cal_lgb"]; self._cal_rf = data["cal_rf"]
        self.scaler = data["scaler"]; self.weights = data["weights"]
        self._xgb_explainer = shap.TreeExplainer(self._xgb)
        self._is_fitted = True
        logger.info("ensemble_model_loaded", path=str(path))
        return self

    def feature_importance(self) -> pd.DataFrame:
        """Average normalised importance across all three models."""
        xgb_imp = self._xgb.feature_importances_ / self._xgb.feature_importances_.sum()
        lgb_imp = self._lgb.feature_importances_ / self._lgb.feature_importances_.sum()
        rf_imp  = self._rf.feature_importances_ / self._rf.feature_importances_.sum()
        avg = (self.weights[0] * xgb_imp + self.weights[1] * lgb_imp + self.weights[2] * rf_imp)
        return (
            pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": avg})
            .sort_values("importance", ascending=False)
        )
