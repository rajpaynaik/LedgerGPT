from __future__ import annotations

"""
API routes for ML model management.
Provides endpoints for model training, evaluation, and prediction.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
import structlog

from models.xgboost_signal_model import XGBoostSignalModel
from scripts.train_ml_model import train_model, evaluate_model

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/model", tags=["model"])

# Global model instance
_model: Optional[XGBoostSignalModel] = None


def init_ml_model():
    """Initialize ML model from disk or create new instance."""
    global _model
    try:
        _model = XGBoostSignalModel("models/xgboost_signal.pkl")
        logger.info("ML model loaded")
    except Exception as e:
        logger.warning("Failed to load ML model, initializing new", error=str(e))
        _model = XGBoostSignalModel()


def get_model() -> XGBoostSignalModel:
    """Get global model instance."""
    global _model
    if _model is None:
        init_ml_model()
    return _model


@router.post("/train")
async def train_xgboost_model(tickers: list[str] = None) -> dict:
    """
    Train XGBoost signal prediction model.
    
    Args:
        tickers: List of tickers for training data (default: major stocks)
        
    Returns:
        Training results and metrics
    """
    try:
        if tickers is None:
            tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "META"]
        
        logger.info("Starting model training", tickers=tickers)
        success = train_model(tickers=tickers)
        
        if success:
            # Reload model
            global _model
            _model = XGBoostSignalModel("models/xgboost_signal.pkl")
            
            importance = _model.get_feature_importance()
            
            return {
                "status": "completed",
                "tickers": tickers,
                "feature_importance": importance,
                "message": "Model trained successfully and saved to models/xgboost_signal.pkl"
            }
        else:
            raise Exception("Training returned False")
    
    except Exception as e:
        logger.error("Model training failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/evaluate")
async def evaluate_xgboost_model(ticker: str = "AAPL") -> dict:
    """
    Evaluate ML model performance on recent data.
    
    Args:
        ticker: Stock ticker for evaluation
        
    Returns:
        Evaluation metrics (accuracy, per-class accuracy)
    """
    try:
        model = get_model()
        metrics = evaluate_model(model, ticker)
        
        return {
            "ticker": ticker,
            "status": "evaluated",
            "metrics": metrics
        }
    
    except Exception as e:
        logger.error("Model evaluation failed", ticker=ticker, error=str(e))
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/predict")
async def predict_signal(
    features: list[float],
    return_proba: bool = False
) -> dict:
    """
    Predict trading signal from feature vector.
    
    Args:
        features: [momentum, rsi, ma, volume, sentiment, volatility]
        return_proba: Include probability distribution
        
    Returns:
        BUY/SELL/HOLD signal with confidence
    """
    try:
        if len(features) != 6:
            raise ValueError("Expected 6 features (momentum, rsi, ma, volume, sentiment, volatility)")
        
        model = get_model()
        signal = model.predict_signal(features)
        
        result = {"signal": signal}
        
        if return_proba:
            probs = model.predict_proba(features)
            result["probabilities"] = probs
        
        return result
    
    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/feature-importance")
async def get_feature_importance() -> dict:
    """
    Get feature importance scores from trained model.
    
    Returns:
        Dictionary of feature names and importance scores
    """
    try:
        model = get_model()
        importance = model.get_feature_importance()
        
        return {
            "status": "success",
            "features": importance,
            "total": sum(importance.values())
        }
    
    except Exception as e:
        logger.error("Failed to get feature importance", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@router.get("/status")
async def get_model_status() -> dict:
    """
    Get current model status and information.
    
    Returns:
        Model status and metadata
    """
    try:
        model = get_model()
        
        return {
            "status": "ready" if model.model else "not_initialized",
            "feature_names": model.feature_names,
            "importance": model.get_feature_importance(),
            "model_path": model.model_path
        }
    
    except Exception as e:
        logger.error("Failed to get model status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")
