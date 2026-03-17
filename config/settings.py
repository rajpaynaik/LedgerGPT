from __future__ import annotations

"""
Centralised settings loaded from environment / .env file.
All other modules import from here — never import os.environ directly.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ─────────────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    secret_key: SecretStr = Field(default="change-me")

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://ledger:ledger@localhost:5432/ledgergpt"
    timescale_enabled: bool = True

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Kafka ────────────────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_raw_social: str = "raw.social"
    kafka_topic_raw_news: str = "raw.news"
    kafka_topic_sentiment: str = "processed.sentiment"
    kafka_topic_signals: str = "trading.signals"
    kafka_group_id: str = "ledgergpt-consumer"

    # ── Twitter/X ───────────────────────────────────────────────────────────
    twitter_bearer_token: SecretStr = Field(default="")
    twitter_api_key: SecretStr = Field(default="")
    twitter_api_secret: SecretStr = Field(default="")
    twitter_access_token: SecretStr = Field(default="")
    twitter_access_secret: SecretStr = Field(default="")

    # ── Reddit ───────────────────────────────────────────────────────────────
    reddit_client_id: str = ""
    reddit_client_secret: SecretStr = Field(default="")
    reddit_user_agent: str = "LedgerGPT/1.0"

    # ── News API ─────────────────────────────────────────────────────────────
    newsapi_key: SecretStr = Field(default="")

    # ── Market Data ──────────────────────────────────────────────────────────
    polygon_api_key: SecretStr = Field(default="")
    alpaca_api_key: SecretStr = Field(default="")
    alpaca_secret_key: SecretStr = Field(default="")
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # ── FinLLaMA ─────────────────────────────────────────────────────────────
    finllama_model_id: str = "FinLLaMA/FinLLaMA-3-8B"
    finllama_device: str = "cuda"
    finllama_max_new_tokens: int = 256
    finllama_batch_size: int = 8
    finllama_quantize: bool = True

    # ── ML Model ─────────────────────────────────────────────────────────────
    model_artifact_path: str = "./artifacts/signal_model.pkl"
    feature_store_path: str = "./artifacts/features/"
    retrain_schedule_hours: int = 24

    # ── Execution ────────────────────────────────────────────────────────────
    broker: str = "alpaca"
    paper_trading: bool = True
    max_position_size_usd: float = 10_000.0
    risk_per_trade_pct: float = 0.02

    # ── Monitoring ───────────────────────────────────────────────────────────
    prometheus_port: int = 9090
    sentry_dsn: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
