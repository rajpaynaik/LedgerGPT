"""
Celery tasks for scheduled and event-driven model retraining.
Schedule via Celery Beat:
  celery -A models.celery_tasks worker --loglevel=info
  celery -A models.celery_tasks beat --loglevel=info
"""
from datetime import datetime, timezone

import structlog
from celery import Celery
from celery.schedules import crontab

from config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Celery app ────────────────────────────────────────────────────────────────
celery_app = Celery(
    "ledgergpt",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# ── Beat schedule ─────────────────────────────────────────────────────────────
celery_app.conf.beat_schedule = {
    # Full retrain every day at 02:00 UTC (after market close)
    "daily-retrain": {
        "task": "models.celery_tasks.retrain_signal_model",
        "schedule": crontab(hour=2, minute=0),
        "args": (None, False),
    },
    # Tuned retrain every Sunday at 03:00 UTC
    "weekly-tuned-retrain": {
        "task": "models.celery_tasks.retrain_signal_model",
        "schedule": crontab(hour=3, minute=0, day_of_week="sunday"),
        "args": (None, True),
    },
    # Prediction sweep every 30 minutes during market hours
    "live-signals": {
        "task": "models.celery_tasks.generate_live_signals",
        "schedule": crontab(minute="*/30", hour="13-22"),  # 09:00-18:00 ET in UTC
    },
    # Online learning update every 6 hours
    "online-update": {
        "task": "models.celery_tasks.online_model_update",
        "schedule": crontab(minute=0, hour="*/6"),
    },
}


# ── Tasks ─────────────────────────────────────────────────────────────────────
@celery_app.task(bind=True, name="models.celery_tasks.retrain_signal_model", max_retries=2)
def retrain_signal_model(self, tickers: list[str] | None = None, tune: bool = False):
    """Full model retrain from fresh data. Replaces artifact on disk."""
    try:
        from models.train import ModelTrainer
        from monitoring.metrics import MODEL_LAST_TRAINED

        logger.info("celery_retrain_starting", tickers=tickers, tune=tune)
        trainer = ModelTrainer()
        model = trainer.retrain(tickers=tickers)
        MODEL_LAST_TRAINED.set_to_current_time()
        logger.info("celery_retrain_complete")
        return {"status": "ok", "trained_at": datetime.now(timezone.utc).isoformat()}
    except Exception as exc:
        logger.error("celery_retrain_failed", error=str(exc))
        raise self.retry(exc=exc, countdown=300)


@celery_app.task(bind=True, name="models.celery_tasks.generate_live_signals", max_retries=1)
def generate_live_signals(self):
    """Generate and publish BUY/HOLD/SELL signals for all watched tickers."""
    try:
        from models.predict import SignalPredictor
        predictor = SignalPredictor()
        signals = predictor.predict_all()
        logger.info("celery_signals_generated", count=len(signals))
        return {"status": "ok", "signals": len(signals)}
    except Exception as exc:
        logger.error("celery_signals_failed", error=str(exc))
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, name="models.celery_tasks.online_model_update", max_retries=1)
def online_model_update(self):
    """
    Incremental online learning: update model with most recent validated signals.
    Uses only recent data to adapt quickly to regime changes without full retrain.
    """
    try:
        from models.train import ModelTrainer
        from ingestion.market_data_ingestion import WATCHLIST

        trainer = ModelTrainer()
        # Retrain on 3 months of recent data — fast update
        X, y = trainer.prepare_data(tickers=WATCHLIST[:6], period="3mo")
        if len(X) < 100:
            logger.warning("online_update_skipped_insufficient_data", rows=len(X))
            return {"status": "skipped", "reason": "insufficient_data"}

        model = trainer.train(X, y, tune_hyperparams=False)
        logger.info("online_update_complete", rows=len(X))
        return {"status": "ok", "rows_trained": len(X)}
    except Exception as exc:
        logger.error("online_update_failed", error=str(exc))
        raise self.retry(exc=exc, countdown=120)


@celery_app.task(name="models.celery_tasks.run_backtest_report")
def run_backtest_report(tickers: list[str] | None = None, period: str = "1y"):
    """Run a backtest and save results — call manually or on a weekly schedule."""
    from backtest.backtester import Backtester
    import json
    from pathlib import Path

    backtester = Backtester()
    results = backtester.run(tickers=tickers or ["TSLA", "AAPL", "NVDA"], period=period)
    out_path = Path("artifacts/backtest_latest.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("backtest_report_saved", path=str(out_path))
    return {"status": "ok", "path": str(out_path)}
