"""
Train the signal model from scratch.
Usage: python -m scripts.train_model [--tickers TSLA AAPL] [--tune]
"""
import argparse
import structlog

from models.train import ModelTrainer

logger = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LedgerGPT signal model")
    parser.add_argument("--tickers", nargs="+", default=None, help="Ticker symbols to train on")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--period", default="2y", help="Historical data period (e.g. 2y, 6mo)")
    args = parser.parse_args()

    trainer = ModelTrainer()
    logger.info("training_started", tickers=args.tickers, tune=args.tune)

    X, y = trainer.prepare_data(tickers=args.tickers, period=args.period)
    model = trainer.train(X, y, tune_hyperparams=args.tune)

    # Feature importance report
    importance = model.feature_importance()
    print("\nTop 15 Feature Importances:")
    print(importance.head(15).to_string(index=False))

    logger.info("training_complete")


if __name__ == "__main__":
    main()
