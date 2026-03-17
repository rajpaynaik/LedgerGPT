-- Sentiment Analysis Storage Schema
-- Stores FinLLM sentiment analysis results for trading signals

CREATE TABLE IF NOT EXISTS sentiment_analysis (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    sentiment_score FLOAT NOT NULL CHECK (sentiment_score BETWEEN -1 AND 1),
    confidence FLOAT NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    source VARCHAR(50) NOT NULL, -- 'twitter', 'reddit', 'news', 'api'
    text_content TEXT,
    event_type VARCHAR(50), -- 'earnings', 'product', 'macro', 'rumor', 'general'
    event_impact VARCHAR(50), -- 'short_term', 'long_term'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast ticker + recent lookups
CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_time 
ON sentiment_analysis(ticker, created_at DESC);

-- Index for cache key lookups
CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_latest 
ON sentiment_analysis(ticker, created_at DESC) 
WHERE created_at > NOW() - INTERVAL '24 hours';

-- Table for tracking sentiment trends
CREATE TABLE IF NOT EXISTS sentiment_metrics (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    avg_sentiment FLOAT,
    sentiment_volume INT, -- number of mentions
    sentiment_velocity FLOAT, -- rate of change
    bullish_count INT,
    bearish_count INT,
    neutral_count INT,
    hour TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sentiment_metrics_ticker_hour 
ON sentiment_metrics(ticker, hour DESC);
