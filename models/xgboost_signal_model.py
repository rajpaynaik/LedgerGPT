from __future__ import annotations

"""
XGBoost-based Signal Generation Model
Replaces manual weighted formula with machine learning for better signal accuracy.
"""

import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import structlog
from xgboost import XGBClassifier

logger = structlog.get_logger(__name__)


class XGBoostSignalModel:
    """
    ML-based signal predictor using XGBoost classifier.
    Learns optimal weights from historical price data and signals.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize XGBoost signal model.
        
        Args:
            model_path: Path to saved model (if exists)
        """
        self.model: Optional[XGBClassifier] = None
        self.feature_names = [
            'price_momentum', 'rsi', 'moving_average', 'volume_spike',
            'sentiment', 'volatility'
        ]
        self.model_path = model_path or "models/xgboost_signal.pkl"
        
        # Try to load existing model
        if model_path and Path(model_path).exists():
            self.load(model_path)
        else:
            self._init_model()
    
    def _init_model(self):
        """Initialize new XGBoost model."""
        self.model = XGBClassifier(
            objective='multi:softprob',  # 3-class: BUY (0), HOLD (1), SELL (2)
            num_class=3,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        logger.info("XGBoost model initialized")
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """
        Train the XGBoost model on historical data.
        
        Args:
            X: Feature vectors (n_samples, 6 features)
            y: Target signals (0=BUY, 1=HOLD, 2=SELL)
            validation_split: Fraction for validation set
        """
        if len(X) < 50:
            logger.warning("Insufficient training data", samples=len(X))
            return False
        
        try:
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            val_score = self.model.score(X_val, y_val)
            
            logger.info(
                "XGBoost model trained",
                samples=len(X),
                train_accuracy=round(train_score, 3),
                val_accuracy=round(val_score, 3)
            )
            
            return True
        
        except Exception as e:
            logger.error("Training failed", error=str(e))
            return False
    
    def predict_proba(self, features: list[float]) -> dict[str, float]:
        """
        Predict signal probabilities for feature vector.
        
        Args:
            features: [momentum, rsi, ma, volume, sentiment, volatility]
            
        Returns:
            Dict with BUY, HOLD, SELL probabilities
        """
        if not self.model:
            return {"BUY": 0.33, "HOLD": 0.34, "SELL": 0.33}
        
        try:
            X = np.array(features).reshape(1, -1)
            probs = self.model.predict_proba(X)[0]
            
            return {
                "BUY": round(float(probs[0]), 3),
                "HOLD": round(float(probs[1]), 3),
                "SELL": round(float(probs[2]), 3)
            }
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            return {"BUY": 0.33, "HOLD": 0.34, "SELL": 0.33}
    
    def predict_signal(self, features: list[float]) -> str:
        """
        Predict single signal (BUY/HOLD/SELL).
        
        Args:
            features: Feature vector
            
        Returns:
            Signal string
        """
        if not self.model:
            return "HOLD"
        
        try:
            X = np.array(features).reshape(1, -1)
            pred = self.model.predict(X)[0]
            signals = ["BUY", "HOLD", "SELL"]
            return signals[int(pred)]
        except Exception:
            return "HOLD"
    
    def get_feature_importance(self) -> dict[str, float]:
        """Get XGBoost feature importance scores."""
        if not self.model:
            return {name: 1.0/6.0 for name in self.feature_names}
        
        try:
            importances = self.model.feature_importances_
            return {
                name: round(float(imp), 3)
                for name, imp in zip(self.feature_names, importances)
            }
        except Exception:
            return {name: 1.0/6.0 for name in self.feature_names}
    
    def save(self, path: str = None):
        """Save model to disk."""
        path = path or self.model_path
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info("Model saved", path=path)
            return True
        except Exception as e:
            logger.error("Failed to save model", error=str(e))
            return False
    
    def load(self, path: str):
        """Load model from disk."""
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded", path=path)
            return True
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            return False
