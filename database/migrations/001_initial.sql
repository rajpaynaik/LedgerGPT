-- LedgerGPT initial schema migration
-- Compatible with PostgreSQL 15+ and TimescaleDB 2.x

-- ── Extensions ──────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ── Sentiment Records ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sentiment_records (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id       VARCHAR(128)   NOT NULL,
    source          VARCHAR(50)    NOT NULL,
    ticker          VARCHAR(10),
    all_tickers     JSONB          DEFAULT '[]',
    sentiment       VARCHAR(10),
    confidence      FLOAT,
    impact          VARCHAR(30),
    reason          TEXT,
    event_type      VARCHAR(40),
    urgency         VARCHAR(10),
    affected_sectors JSONB         DEFAULT '[]',
    raw_text        TEXT,
    processed_at    TIMESTAMPTZ    NOT NULL,
    created_at      TIMESTAMPTZ    DEFAULT NOW(),
    CONSTRAINT uq_sentiment_source_id UNIQUE (source_id, source)
);

-- Convert to TimescaleDB hypertable (partition by processed_at, 7 day chunks)
SELECT create_hypertable(
    'sentiment_records',
    'processed_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS ix_sentiment_ticker_time
    ON sentiment_records (ticker, processed_at DESC);

-- Compression policy (compress chunks older than 30 days)
ALTER TABLE sentiment_records SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker'
);
SELECT add_compression_policy('sentiment_records', INTERVAL '30 days', if_not_exists => TRUE);

-- ── Trading Signals ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trading_signals (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker          VARCHAR(10)    NOT NULL,
    signal          VARCHAR(5)     NOT NULL,
    confidence      FLOAT,
    prob_buy        FLOAT,
    prob_hold       FLOAT,
    prob_sell       FLOAT,
    reason          TEXT,
    top_factors     JSONB          DEFAULT '[]',
    generated_at    TIMESTAMPTZ    NOT NULL,
    executed        BOOLEAN        DEFAULT FALSE
);

SELECT create_hypertable(
    'trading_signals',
    'generated_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS ix_signal_ticker_time
    ON trading_signals (ticker, generated_at DESC);

-- ── Orders ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orders (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    broker_order_id  VARCHAR(128)   UNIQUE,
    ticker           VARCHAR(10)    NOT NULL,
    side             VARCHAR(5),
    qty              FLOAT,
    price_at_signal  FLOAT,
    fill_price       FLOAT,
    status           VARCHAR(20),
    signal_confidence FLOAT,
    signal_reason    TEXT,
    submitted_at     TIMESTAMPTZ,
    filled_at        TIMESTAMPTZ,
    pnl              FLOAT
);

CREATE INDEX IF NOT EXISTS ix_orders_ticker ON orders (ticker);
CREATE INDEX IF NOT EXISTS ix_orders_submitted ON orders (submitted_at DESC);

-- ── Price Bars ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS price_bars (
    id          BIGSERIAL,
    ticker      VARCHAR(10)    NOT NULL,
    timestamp   TIMESTAMPTZ    NOT NULL,
    open        FLOAT,
    high        FLOAT,
    low         FLOAT,
    close       FLOAT,
    volume      FLOAT,
    vwap        FLOAT,
    timeframe   VARCHAR(5)     DEFAULT '1d',
    CONSTRAINT uq_price_bar UNIQUE (ticker, timestamp, timeframe)
);

SELECT create_hypertable(
    'price_bars',
    'timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS ix_price_ticker_time
    ON price_bars (ticker, timestamp DESC);

ALTER TABLE price_bars SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker,timeframe'
);
SELECT add_compression_policy('price_bars', INTERVAL '90 days', if_not_exists => TRUE);

-- ── Continuous Aggregate: hourly sentiment ────────────────────────────────
CREATE MATERIALIZED VIEW IF NOT EXISTS sentiment_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', processed_at) AS bucket,
    ticker,
    COUNT(*)                             AS record_count,
    AVG(confidence)                      AS avg_confidence,
    SUM(CASE WHEN sentiment = 'bullish' THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(COUNT(*), 0)              AS bullish_ratio,
    SUM(CASE WHEN sentiment = 'bearish' THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(COUNT(*), 0)              AS bearish_ratio
FROM sentiment_records
WHERE ticker IS NOT NULL
GROUP BY bucket, ticker
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'sentiment_hourly',
    start_offset    => INTERVAL '3 hours',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);

-- ── Retention policies ────────────────────────────────────────────────────
SELECT add_retention_policy('sentiment_records', INTERVAL '90 days',  if_not_exists => TRUE);
SELECT add_retention_policy('trading_signals',   INTERVAL '365 days', if_not_exists => TRUE);
SELECT add_retention_policy('price_bars',        INTERVAL '5 years',  if_not_exists => TRUE);
