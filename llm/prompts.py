"""
Prompt templates for FinLLaMA tasks.
All prompts follow the instruction-tuning format expected by FinLLaMA-3.
"""
from string import Template

# ── Sentiment Analysis ──────────────────────────────────────────────────────
SENTIMENT_PROMPT = Template("""You are a financial analyst AI. Analyse the following text and return a JSON object.

Text: "$text"

Respond ONLY with a valid JSON object in this exact format:
{
  "ticker": "<stock ticker symbol or null>",
  "sentiment": "<bullish|bearish|neutral>",
  "confidence": <float 0.0-1.0>,
  "impact": "<short_term_positive|short_term_negative|long_term_positive|long_term_negative|neutral>",
  "reason": "<one-sentence explanation>"
}""")

# ── Multi-ticker extraction ─────────────────────────────────────────────────
TICKER_EXTRACTION_PROMPT = Template("""Extract all stock ticker symbols mentioned in the following financial text.
Return a JSON array of uppercase ticker symbols only. If none found, return [].

Text: "$text"

Response (JSON array only):""")

# ── Event classification ────────────────────────────────────────────────────
EVENT_CLASSIFICATION_PROMPT = Template("""Classify the market event described in the following text.

Text: "$text"

Choose one of: earnings_beat, earnings_miss, product_launch, merger_acquisition,
regulatory_action, insider_trading, analyst_upgrade, analyst_downgrade,
macro_economic, geopolitical, general_news

Respond ONLY with a JSON object:
{
  "event_type": "<classification>",
  "urgency": "<high|medium|low>",
  "affected_sectors": ["<sector1>", "<sector2>"]
}""")

# ── Batch sentiment ─────────────────────────────────────────────────────────
BATCH_SENTIMENT_SYSTEM = """You are a financial sentiment analysis engine.
For each text in the input array, return structured sentiment analysis.
Always respond with a JSON array matching the input order."""

BATCH_SENTIMENT_USER = Template("""Analyse sentiment for each of the following financial texts.
Return a JSON array where each element has: ticker, sentiment, confidence, impact, reason.

Texts:
$texts

Response (JSON array only):""")


def build_sentiment_prompt(text: str) -> str:
    return SENTIMENT_PROMPT.substitute(text=text.replace('"', "'"))


def build_ticker_prompt(text: str) -> str:
    return TICKER_EXTRACTION_PROMPT.substitute(text=text.replace('"', "'"))


def build_event_prompt(text: str) -> str:
    return EVENT_CLASSIFICATION_PROMPT.substitute(text=text.replace('"', "'"))


def build_batch_sentiment_prompt(texts: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    return BATCH_SENTIMENT_USER.substitute(texts=numbered)
