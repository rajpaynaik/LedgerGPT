# LedgerGPT — AI Trading Intelligence System

A production-grade AI trading system powered by **FinLLaMA** for financial sentiment analysis and **XGBoost** for quantitative signal generation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                        │
│  Twitter/X ──┐                                                  │
│  Reddit ─────┼──► Kafka Topics ──► Deduplication Filter        │
│  News API ───┘     (raw.social,      (Redis bloom)             │
│  Market Data       raw.news)                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   LLM SENTIMENT ENGINE                           │
│  FinLLaMA-3-8B (4-bit quantised, single A100/RTX 3090)         │
│  • Sentiment scoring (bullish/bearish/neutral)                  │
│  • Ticker extraction                                            │
│  • Event classification                                         │
│  • Confidence scoring                                           │
│  Output ──► Kafka: processed.sentiment                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                 FEATURE ENGINEERING LAYER                        │
│  Sentiment features + Technical indicators (RSI, MACD, BB,     │
│  Volume Spike, Momentum, Volatility, ADX, EMA crossovers)      │
│  → 32-dimension feature vector per ticker                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   ML DECISION MODEL                              │
│  XGBoost Multi-class Classifier                                 │
│  • Calibrated probabilities (Platt/Isotonic)                    │
│  • SHAP explainability                                          │
│  • Optuna hyperparameter optimisation                           │
│  Output: BUY / HOLD / SELL + confidence score                   │
│  ──► Kafka: trading.signals                                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│               TRADING EXECUTION LAYER (Optional)                 │
│  OrderManager → Alpaca Markets (Paper/Live)                     │
│  • Kelly fraction position sizing                               │
│  • Stop-loss / take-profit enforcement                          │
│  • Risk checks before every order                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone and configure

```bash
cp .env.example .env
# Fill in your API keys
```

### 2. Start infrastructure

```bash
docker-compose up -d timescaledb redis kafka
```

### 3. Apply DB schema

```bash
docker-compose exec timescaledb psql -U ledger -d ledgergpt \
  -f /docker-entrypoint-initdb.d/001_initial.sql
```

### 4. Train the model

```bash
pip install -r requirements.txt
python -m scripts.train_model --tickers TSLA AAPL NVDA MSFT --period 2y
```

### 5. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start ingestion + sentiment workers

```bash
# Terminal 1
python -m scripts.run_ingestion

# Terminal 2
python -m scripts.run_sentiment_worker
```

### 7. Full stack with Docker

```bash
docker-compose up -d
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/sentiment/analyse` | Analyse text with FinLLaMA |
| POST | `/api/v1/sentiment/batch` | Batch sentiment (up to 50 texts) |
| GET | `/api/v1/signals/` | Live signals for all tickers |
| GET | `/api/v1/signals/{ticker}` | Signal for a specific ticker |
| POST | `/api/v1/signals/backtest` | Run historical backtest |
| GET | `/metrics` | Prometheus metrics |

**API Docs:** http://localhost:8000/docs

---

## Example Usage

### Sentiment Analysis

```python
import httpx

response = httpx.post("http://localhost:8000/api/v1/sentiment/analyse", json={
    "text": "Tesla deliveries beat expectations for Q4, strong momentum heading into 2025"
})
print(response.json())
# {
#   "ticker": "TSLA",
#   "sentiment": "bullish",
#   "confidence": 0.84,
#   "impact": "short_term_positive",
#   "reason": "positive delivery beat suggests strong demand",
#   "event_type": "earnings_beat"
# }
```

### Get Trading Signal

```python
response = httpx.get("http://localhost:8000/api/v1/signals/TSLA")
print(response.json())
# {
#   "ticker": "TSLA",
#   "signal": "BUY",
#   "confidence": 0.74,
#   "reason": "rsi_14 is positive (SHAP=0.142)",
#   "top_factors": [...]
# }
```

### Run Backtest

```bash
python -m scripts.run_backtest --tickers TSLA AAPL NVDA --period 2y --cash 100000
```

---

## Project Structure

```
LedgerGPT/
├── config/              # Settings (pydantic-settings)
├── ingestion/           # Twitter, Reddit, News, Market data
├── stream/              # Kafka producer/consumer, Redis dedup
├── llm/                 # FinLLaMA service, Sentiment engine
├── features/            # Technical indicators, Sentiment aggregator
├── models/              # XGBoost signal model, Training, Prediction
├── backtest/            # Backtester, Strategy, Metrics
├── execution/           # Broker API (Alpaca), Order Manager
├── database/            # SQLAlchemy models, CRUD, TimescaleDB migrations
├── api/                 # FastAPI app + routes
├── monitoring/          # Prometheus metrics, Alert manager
├── scripts/             # CLI entry points
└── tests/               # Unit tests
```

---

## GPU Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| FinLLaMA-3-8B (4-bit) | 8 GB VRAM | 16 GB (RTX 3080/A10) |
| FinLLaMA-3-8B (full) | 20 GB VRAM | A100 40GB |
| XGBoost training | CPU only | CPU (fast) |

To run without GPU, set `FINLLAMA_DEVICE=cpu` in `.env` (inference will be slower).

---

## Monitoring

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin / ledgergpt)
- **Kafka UI**: http://localhost:8080

---

## Design Principles

> FinLLaMA is **not** the decision-maker.
> It extracts structured signal features from unstructured text.
> The XGBoost model makes the final BUY/HOLD/SELL determination.

- **Reliability**: Kafka for durability, Redis for dedup, retry logic everywhere
- **Explainability**: SHAP values for every signal
- **Backtestability**: Exact model replay on historical data via Backtrader
- **Modularity**: Each layer is independently testable and deployable

---

## License

MIT
