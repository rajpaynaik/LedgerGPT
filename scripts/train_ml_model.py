from __future__ import annotations

"""
Training script for XGBoost signal model.
Generates synthetic training data from historical market patterns and trains the model.
"""

import json
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf
import structlog
from xgboost import XGBClassifier

from models.xgboost_signal_model import XGBoostSignalModel
from models.predict import SignalPredictor

logger = structlog.get_logger(__name__)


def generate_training_data(
    ticker: str = "AAPL",
    period: str = "2y",
    n_samples: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate training data from historical price data.
    Creates feature vectors and labels based on price movements.
    
    Args:
        ticker: Stock ticker
        period: Historical period ('1y', '2y', '5y')
        n_samples: Number of training samples to generate
        
    Returns:
        (X, y) where X is features array, y is signal labels
    """
    try:
        # Fetch historical data
        df = yf.download(ticker, period=period, progress=False)
        
        if len(df) < 100:
            logger.warning("Insufficient data", ticker=ticker, rows=len(df))
            return np.array([]), np.array([])
        
        # Calculate returns (label: 1=BUY if +2% next day, 2=SELL if -2%, else 0=HOLD)
        df['returns'] = df['Close'].pct_change()
        df['signal'] = 0  # HOLD
        df.loc[df['returns'] > 0.02, 'signal'] = 0   # BUY
        df.loc[df['returns'] < -0.02, 'signal'] = 2  # SELL
        df.loc[(df['returns'] >= -0.02) & (df['returns'] <= 0.02), 'signal'] = 1  # HOLD
        
        # Shift signal to align with features (we predict next day's signal)
        df['signal'] = df['signal'].shift(-1)
        df = df.dropna()
        
        # Calculate features (same as in predict.py)
        features = []
        labels = []
        
        for i in range(len(df) - 1):
            try:
                window = df.iloc[max(0, i-20):i+1]
                
                if len(window) < 5:
                    continue
                
                # Feature 1: Price Momentum (20-day)
                close_values = window['Close'].values
                momentum = float((close_values[-1] - close_values[0]) / close_values[0])
                
                # Feature 2: RSI (14-day)
                deltas = window['Close'].diff().dropna().values
                if len(deltas) >= 14:
                    seed = deltas[:14]
                    up = float(seed[seed >= 0].sum()) / 14
                    down = float(-seed[seed < 0].sum()) / 14
                    rs = up / down if down != 0 else 1.0
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                else:
                    rsi = 50.0
                
                # Feature 3: Moving Average (distance to 50-day MA)
                ma50 = float(window['Close'].mean())
                current_close = float(window['Close'].iloc[-1])
                ma_signal = (current_close - ma50) / ma50 if ma50 != 0 else 0.0
                
                # Feature 4: Volume Spike
                volume_values = window['Volume'].values
                avg_volume = float(np.mean(volume_values[:-1])) if len(volume_values) > 1 else 1.0
                volume_spike = float(volume_values[-1]) / avg_volume if avg_volume > 0 else 1.0
                
                # Feature 5: Sentiment (use proxy: positive momentum)
                sentiment = min(1.0, max(-1.0, float(momentum))) / 2 + 0.5
                
                # Feature 6: Volatility
                pct_changes = window['Close'].pct_change().dropna().values
                volatility = float(np.std(pct_changes)) if len(pct_changes) > 0 else 0.01
                
                features.append([
                    float(momentum), float(rsi / 100), float(ma_signal), float(volume_spike), float(sentiment), float(volatility)
                ])
                
                signal_val = int(df['signal'].iloc[i+1])
                labels.append(signal_val)
                
                if len(features) >= n_samples:
                    break
            
            except Exception as e:
                logger.debug("Feature calculation error", error=str(e), index=i)
                continue
        
        X = np.array(features)
        y = np.array(labels)
        
        logger.info(
            "Training data generated",
            ticker=ticker,
            samples=len(X),
            buy_count=sum(y == 0),
            hold_count=sum(y == 1),
            sell_count=sum(y == 2)
        )
        
        return X, y
    
    except Exception as e:
        logger.error("Data generation failed", error=str(e))
        return np.array([]), np.array([])


def train_model(
    output_path: str = "models/xgboost_signal.pkl",
    tickers: list[str] = None,
    period: str = "2y"
) -> bool:
    """
    Train XGBoost signal model on multiple tickers.
    
    Args:
        output_path: Path to save trained model
        tickers: List of tickers for training data
        period: Historical period for data
        
    Returns:
        True if training succeeded
    """
    if not tickers:
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "META"]
    
    try:
        # Generate combined training data
        X_all, y_all = [], []
        
        for ticker in tickers:
            X, y = generate_training_data(ticker, period)
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
        
        if not X_all:
            logger.error("No training data generated")
            return False
        
        X_combined = np.vstack(X_all)
        y_combined = np.hstack(y_all)
        
        logger.info(
            "Training on combined data",
            total_samples=len(X_combined),
            tickers=len(tickers)
        )
        
        # Train model
        model = XGBoostSignalModel(output_path)
        success = model.train(X_combined, y_combined)
        
        if success:
            model.save(output_path)
            
            # Print feature importance
            importance = model.get_feature_importance()
            logger.info("Feature importance", **importance)
            
            return True
        else:
            return False
    
    except Exception as e:
        logger.error("Training failed", error=str(e))
        return False


def evaluate_model(model: XGBoostSignalModel, ticker: str = "AAPL") -> dict:
    """
    Evaluate model on recent data.
    
    Args:
        model: Trained XGBoostSignalModel
        ticker: Ticker for evaluation
        
    Returns:
        Evaluation metrics
    """
    try:
        X, y = generate_training_data(ticker, period="6mo", n_samples=100)
        
        if len(X) == 0:
            return {}
        
        # Predict on test data
        predictions = np.array([model.predict_signal(x) for x in X])
        signal_map = {"BUY": 0, "HOLD": 1, "SELL": 2}
        y_pred = np.array([signal_map[p] for p in predictions])
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y)
        
        # Per-class accuracy
        buy_acc = np.mean(y_pred[y == 0] == y[y == 0]) if sum(y == 0) > 0 else 0
        hold_acc = np.mean(y_pred[y == 1] == y[y == 1]) if sum(y == 1) > 0 else 0
        sell_acc = np.mean(y_pred[y == 2] == y[y == 2]) if sum(y == 2) > 0 else 0
        
        return {
            "accuracy": round(accuracy, 3),
            "buy_accuracy": round(buy_acc, 3),
            "hold_accuracy": round(hold_acc, 3),
            "sell_accuracy": round(sell_acc, 3),
            "samples": len(X)
        }
    
    except Exception as e:
        logger.error("Evaluation failed", error=str(e))
        return {}


if __name__ == "__main__":
    import sys
    
    action = sys.argv[1] if len(sys.argv) > 1 else "train"
    
    if action == "train":
        print("Starting XGBoost model training...")
        success = train_model()
        if success:
            print("✅ Model trained successfully (models/xgboost_signal.pkl)")
        else:
            print("❌ Training failed")
    
    elif action == "evaluate":
        print("Evaluating model...")
        model = XGBoostSignalModel("models/xgboost_signal.pkl")
        metrics = evaluate_model(model)
        if metrics:
            print(f"Evaluation metrics: {json.dumps(metrics, indent=2)}")
        else:
            print("Evaluation failed")
    
    else:
        print(f"Unknown action: {action}")
        print("Usage: python train_ml_model.py [train|evaluate]")
