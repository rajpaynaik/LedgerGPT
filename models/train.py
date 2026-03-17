from __future__ import annotations

"""
Model training pipeline with Optuna hyperparameter optimisation,
cross-validation, and MLflow-style experiment tracking.
"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import structlog
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
import xgboost as xgb

from config import get_settings
from features.feature_engineering import FEATURE_COLUMNS, FeatureEngineer
from ingestion.market_data_ingestion import MarketDataIngester, WATCHLIST
from .signal_model import SignalModel, LABEL_MAP

logger = structlog.get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    """Orchestrates full train / tune / evaluate / save cycle."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.feature_eng = FeatureEngineer()
        self.market_data = MarketDataIngester()

    # ── Data preparation ───────────────────────────────────────────────────
    def prepare_data(
        self,
        tickers: list[str] | None = None,
        sentiment_records: list[dict] | None = None,
        period: str = "2y",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Download price data, compute features, generate labels.
        Returns (X, y) ready for model training.
        """
        tickers = tickers or WATCHLIST[:6]
        sentiment_records = sentiment_records or []

        all_frames = []
        for ticker in tickers:
            try:
                price_df = self.market_data.fetch_ohlcv_yf(ticker, period=period)
                feat_df = self.feature_eng.build_training_dataset(
                    price_df, sentiment_records, ticker
                )
                feat_df["ticker"] = ticker
                all_frames.append(feat_df)
                self.feature_eng.save_features(feat_df, ticker)
            except Exception as exc:
                logger.error("data_prep_error", ticker=ticker, error=str(exc))

        if not all_frames:
            raise ValueError("No training data could be prepared.")

        combined = pd.concat(all_frames, axis=0).sort_index()
        y = combined["label"].astype(int)
        X = combined.drop(columns=["label", "ticker"], errors="ignore")

        logger.info(
            "training_data_prepared",
            rows=len(X),
            tickers=tickers,
            label_dist=y.value_counts().to_dict(),
        )
        return X, y

    # ── Optuna hyperparameter search ───────────────────────────────────────
    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": 42,
            "n_jobs": -1,
        }

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        X_feat = X[FEATURE_COLUMNS].fillna(0)

        for train_idx, val_idx in tscv.split(X_feat):
            X_tr, X_val = X_feat.iloc[train_idx], X_feat.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = clf.predict(X_val)
            scores.append(f1_score(y_val, preds, average="weighted"))

        return float(np.mean(scores))

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
    ) -> dict:
        """Run Optuna hyperparameter search."""
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        logger.info(
            "hyperparameter_tuning_complete",
            best_score=study.best_value,
            best_params=study.best_params,
        )
        return study.best_params

    # ── Full training run ──────────────────────────────────────────────────
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tune_hyperparams: bool = False,
        val_split: float = 0.2,
    ) -> SignalModel:
        """Train the signal model and return a fitted instance."""
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        params = None
        if tune_hyperparams:
            best_params = self.tune(X_train, y_train, n_trials=30)
            params = {**best_params, **{
                "use_label_encoder": False,
                "eval_metric": "mlogloss",
                "tree_method": "hist",
                "objective": "multi:softprob",
                "num_class": 3,
                "random_state": 42,
                "n_jobs": -1,
            }}

        model = SignalModel(params=params)
        model.fit(X_train, y_train, X_val, y_val)

        # Evaluate
        self.evaluate(model, X_val, y_val)
        model.save()
        return model

    # ── Evaluation ────────────────────────────────────────────────────────
    def evaluate(
        self, model: SignalModel, X: pd.DataFrame, y: pd.Series
    ) -> dict:
        predictions = model.predict(X)
        pred_signals = [p["signal"] for p in predictions]
        pred_labels = [{"BUY": 2, "HOLD": 1, "SELL": 0}[s] for s in pred_signals]

        acc = accuracy_score(y, pred_labels)
        f1 = f1_score(y, pred_labels, average="weighted")
        report = classification_report(
            y, pred_labels, target_names=["SELL", "HOLD", "BUY"]
        )

        logger.info(
            "model_evaluation",
            accuracy=round(acc, 4),
            f1_weighted=round(f1, 4),
        )
        print("\n" + report)

        return {
            "accuracy": acc,
            "f1_weighted": f1,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Retraining ─────────────────────────────────────────────────────────
    def retrain(self, tickers: list[str] | None = None) -> SignalModel:
        """
        Full retrain cycle: fetch fresh data → build features → train → save.
        Called on a schedule (e.g. every 24h via Celery beat).
        """
        logger.info("retraining_started")
        X, y = self.prepare_data(tickers=tickers)
        model = self.train(X, y)
        logger.info("retraining_complete")
        return model
