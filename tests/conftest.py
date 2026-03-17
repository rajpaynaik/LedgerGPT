"""
Pytest configuration and shared fixtures.
"""
import os

import pytest

# Prevent accidental API calls during tests
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("TWITTER_BEARER_TOKEN", "test_token")
os.environ.setdefault("REDDIT_CLIENT_ID", "test_id")
os.environ.setdefault("NEWSAPI_KEY", "test_key")
os.environ.setdefault("POLYGON_API_KEY", "test_key")
os.environ.setdefault("ALPACA_API_KEY", "test_key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test_secret")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://ledger:ledger@localhost:5432/ledgergpt_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
