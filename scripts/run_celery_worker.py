"""
Start the Celery worker + beat scheduler for model retraining.

Usage:
  # Worker only
  python -m scripts.run_celery_worker worker

  # Beat scheduler only (runs in separate process)
  python -m scripts.run_celery_worker beat

  # Both in one process (dev only — not for production)
  python -m scripts.run_celery_worker all
"""
import sys
import structlog

logger = structlog.get_logger(__name__)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "worker"

    from models.celery_tasks import celery_app

    if mode == "worker":
        logger.info("celery_worker_starting")
        celery_app.worker_main(
            argv=["worker", "--loglevel=info", "--concurrency=2", "-Q", "celery"]
        )
    elif mode == "beat":
        logger.info("celery_beat_starting")
        celery_app.start(argv=["beat", "--loglevel=info"])
    elif mode == "all":
        # Dev convenience: embed beat within worker (not recommended for prod)
        logger.warning("celery_embedded_beat_dev_only")
        celery_app.worker_main(
            argv=["worker", "--loglevel=info", "--beat", "--concurrency=1"]
        )
    else:
        print(f"Unknown mode: {mode}. Use: worker | beat | all")
        sys.exit(1)


if __name__ == "__main__":
    main()
