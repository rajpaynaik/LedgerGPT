from __future__ import annotations

"""
Real-time signal predictor.
Consumes processed.sentiment from Kafka, fetches latest price data,
builds inference vectors, and publishes trading.signals.
Now integrated with FinLLM Sentiment Worker for live sentiment analysis.
"""
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import structlog

from config import get_settings
from features.feature_engineering import FeatureEngineer
from ingestion.market_data_ingestion import MarketDataIngester, WATCHLIST
from stream.kafka_producer import KafkaProducerClient
from .signal_model import SignalModel

logger = structlog.get_logger(__name__)


class SignalPredictor:
    """
    Generates live trading signals by combining latest market data
    with real-time sentiment from FinLLM (or fallback to buffer).
    """

    def __init__(self, sentiment_worker = None) -> None:
        self.settings = get_settings()
        self.feature_eng = FeatureEngineer()
        self.market_data = MarketDataIngester()
        self.producer = KafkaProducerClient()
        self._model: SignalModel | None = None
        self._sentiment_buffer: list[dict] = []
        self._buffer_maxlen = 10_000
        
        # Optional: sentiment worker for live FinLLM-based sentiment
        self.sentiment_worker = sentiment_worker

    def _get_model(self) -> SignalModel:
        if self._model is None:
            self._model = SignalModel()
            self._model.load()
        return self._model

    def ingest_sentiment(self, record: dict, _key: str) -> None:
        """Kafka handler — buffers incoming sentiment records."""
        self._sentiment_buffer.append(record)
        if len(self._sentiment_buffer) > self._buffer_maxlen:
            self._sentiment_buffer = self._sentiment_buffer[-self._buffer_maxlen:]

    def predict_ticker(self, ticker: str) -> dict:
        """
        Generate a signal for a single ticker using multiple technical indicators
        and sentiment data with risk filters.
        """
        try:
            # Fetch recent price data
            price_df = self.market_data.fetch_ohlcv_yf(ticker, period="6mo")
            
            if price_df is None or len(price_df) < 50:
                raise ValueError(f"Insufficient price data for {ticker}")
            
            # Extract scalars safely - convert everything to Python float
            latest_price = float(price_df['close'].iloc[-1].item())
            prev_price = float(price_df['close'].iloc[-2].item())
            latest_volume = float(price_df['volume'].iloc[-1].item()) if 'volume' in price_df.columns else 1.0
            avg_volume_val = float(price_df['volume'].tail(20).mean().item()) if 'volume' in price_df.columns else 1.0
            
            # ── Feature 1: Price Momentum ────────────────────────────────
            if prev_price > 0:
                price_change_pct = ((latest_price - prev_price) / prev_price) * 100
            else:
                price_change_pct = 0.0
            momentum_score = min(1.0, abs(price_change_pct) / 5)
            momentum_direction = 1.0 if price_change_pct > 0 else -1.0
            
            # ── Feature 2: Volume Spike ─────────────────────────────────
            if avg_volume_val > 0:
                volume_spike = latest_volume / avg_volume_val
            else:
                volume_spike = 1.0
            volume_score = min(1.0, max(0, (volume_spike - 1.0) / 2))
            
            # ── Feature 3: RSI (Relative Strength Index) ────────────────
            close_prices = price_df['close'].tail(14).values
            rsi_score = 0.5
            if len(close_prices) >= 14:
                deltas = []
                for i in range(1, len(close_prices)):
                    deltas.append(float(close_prices[i]) - float(close_prices[i-1]))
                
                gains = [max(0.0, d) for d in deltas]
                losses = [max(0.0, -d) for d in deltas]
                avg_gain = sum(gains) / len(gains) if len(gains) > 0 else 0.0
                avg_loss = sum(losses) / len(losses) if len(losses) > 0 else 0.0
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                else:
                    rsi = 50.0 if avg_gain == 0 else 100.0
                
                if 30 < rsi < 70:
                    rsi_score = (rsi - 30.0) / 40.0
                elif rsi >= 70:
                    rsi_score = 1.0
                else:
                    rsi_score = 0.0
            
            # ── Feature 4: Moving Average Crossover ──────────────────────
            ma_20 = float(price_df['close'].tail(20).mean().item())
            ma_50 = float(price_df['close'].tail(50).mean().item())
            if ma_50 > 0:
                ma_ratio = (ma_20 - ma_50) / ma_50
            else:
                ma_ratio = 0.0
            ma_score = min(1.0, abs(ma_ratio) * 20.0)
            ma_direction = 1.0 if ma_ratio > 0 else -1.0
            
            # ── Feature 5: Sentiment (from FinLLM worker or buffer) ────────
            sentiment_score = 0.5
            
            # Try to get sentiment from worker first (FinLLM-based)
            if self.sentiment_worker:
                try:
                    sentiment_summary = self.sentiment_worker.get_sentiment_summary(ticker)
                    if sentiment_summary:
                        # Use aggregated sentiment from database
                        sentiment_score = ((sentiment_summary['avg_sentiment'] + 1.0) / 2.0)  # Normalize to 0-1
                        logger.debug("Using FinLLM sentiment for signal", ticker=ticker, score=sentiment_score)
                except Exception as e:
                    logger.debug("Failed to fetch FinLLM sentiment, falling back to buffer", error=str(e))
            
            # Fallback to buffer if no worker or no sentiment data
            if sentiment_score == 0.5 and self._sentiment_buffer and len(self._sentiment_buffer) > 0:
                positive_count = sum(1 for s in self._sentiment_buffer if s.get('sentiment') == 'bullish')
                negative_count = sum(1 for s in self._sentiment_buffer if s.get('sentiment') == 'bearish')
                total_sent = len(self._sentiment_buffer)
                if total_sent > 0:
                    sentiment_score = float(positive_count - negative_count) / float(total_sent)
                    sentiment_score = (sentiment_score + 1.0) / 2.0  # Normalize to 0-1
            
            # ── Feature 6: Volatility ───────────────────────────────────
            returns = price_df['close'].pct_change().tail(20).values.flatten()
            volatility = 0.01
            if len(returns) > 1:
                volatility = float(pd.Series(returns).std())
            volatility_score = min(1.0, volatility * 10.0)
            
            # ── Composite Scoring with Weights ──────────────────────────
            feature_scores = {
                "price_momentum": (momentum_score * momentum_direction, 0.25),
                "volume_spike": (volume_score, 0.15),
                "rsi": (rsi_score, 0.20),
                "moving_average": (ma_score * ma_direction, 0.20),
                "sentiment": ((sentiment_score - 0.5) * 2.0, 0.12),
                "volatility": (volatility_score, 0.08),
            }
            
            # Calculate weighted score
            total_signal = 0.0
            for score, weight in feature_scores.values():
                total_signal += score * weight
            
            # ── Risk Filters ────────────────────────────────────────────
            # More aggressive probability scaling
            buy_prob = 0.25 + max(0.0, total_signal) * 0.50
            sell_prob = 0.25 + max(0.0, -total_signal) * 0.50
            hold_prob = max(0.0, 1.0 - buy_prob - sell_prob)
            
            # Normalize probabilities to sum to 1.0
            total_prob = buy_prob + hold_prob + sell_prob
            if total_prob > 0:
                buy_prob = buy_prob / total_prob
                hold_prob = hold_prob / total_prob
                sell_prob = sell_prob / total_prob
            
            max_prob = max(buy_prob, hold_prob, sell_prob)
            
            # Volume check
            if volume_spike < 0.8:
                total_signal = total_signal * 0.5
            
            # Determine signal FIRST - lowered thresholds for more responsive signals
            if abs(total_signal) < 0.15:
                # Very weak signal
                signal = "HOLD"
                confidence = 0.40
            elif total_signal > 0.2:
                # Bullish
                signal = "BUY"
                confidence = min(0.95, 0.40 + buy_prob * 0.4)
            elif total_signal < -0.2:
                # Bearish
                signal = "SELL"
                confidence = min(0.95, 0.40 + sell_prob * 0.4)
            else:
                # Mixed signals
                signal = "HOLD"
                confidence = max_prob
            
            # ── Generate Detailed Reasoning ────────────────────────────
            def generate_detailed_reason(
                ticker, signal, total_signal, feature_scores,
                price_change_pct, rsi, ma_ratio, volume_spike, 
                sentiment_score, volatility, max_prob, buy_prob, 
                sell_prob, hold_prob, confidence
            ) -> str:
                """
                Generate comprehensive reasoning for the trading signal.
                """
                lines = []
                lines.append(f"=== {ticker} Signal Analysis ===\n")
                
                # Overall market sentiment
                sentiment_direction = "BULLISH" if total_signal > 0 else "BEARISH" if total_signal < 0 else "NEUTRAL"
                lines.append(f"Overall Market Signal: {sentiment_direction} (composite score: {total_signal:.3f})")
                lines.append(f"Signal Confidence: {round(max_prob * 100, 1)}% | BUY: {round(buy_prob*100, 1)}% | HOLD: {round(hold_prob*100, 1)}% | SELL: {round(sell_prob*100, 1)}%\n")
                
                # Feature-by-feature breakdown
                lines.append("📊 TECHNICAL FACTORS BREAKDOWN:")
                
                # Price Momentum (25% weight)
                momentum_desc = "STRONG UPSIDE" if price_change_pct > 5 else "MODERATE UP" if price_change_pct > 1 else "WEAK UP" if price_change_pct > 0 else "FLAT" if price_change_pct > -1 else "MODERATE DOWN" if price_change_pct > -5 else "STRONG DOWNSIDE"
                lines.append(f"  • Price Momentum (25% weight): {momentum_desc} - Daily change: {price_change_pct:+.2f}%")
                
                # RSI (Overbought/Oversold)
                if rsi > 70:
                    rsi_desc = f"OVERBOUGHT ({rsi:.1f}). Potential pullback or consolidation expected."
                elif rsi < 30:
                    rsi_desc = f"OVERSOLD ({rsi:.1f}). Potential bounce or reversal expected."
                else:
                    rsi_desc = f"NEUTRAL ({rsi:.1f}). Normal trading range."
                lines.append(f"  • RSI Indicator (20% weight): {rsi_desc}")
                
                # Moving Average Crossover (20% weight)
                ma_trend = "UPTREND" if ma_ratio > 0 else "DOWNTREND"
                lines.append(f"  • Moving Average Crossover (20% weight): {ma_trend} - 20-day MA {'above' if ma_ratio > 0 else 'below'} 50-day MA by {abs(ma_ratio)*100:.2f}%")
                
                # Volume Spike (15% weight)
                if volume_spike > 1.5:
                    vol_desc = "ELEVATED - Strong trading interest"
                elif volume_spike > 1.0:
                    vol_desc = "NORMAL - Average participation"
                else:
                    vol_desc = f"WEAK ({volume_spike:.2f}x) - Low participation, signals may be less reliable"
                lines.append(f"  • Volume Spike (15% weight): {vol_desc}")
                
                # Sentiment (12% weight)
                sentiment_normalized = (sentiment_score - 0.5) * 2.0  # Convert back to -1 to 1
                if sentiment_normalized > 0.5:
                    sent_desc = "STRONGLY BULLISH"
                elif sentiment_normalized > 0:
                    sent_desc = "BULLISH"
                elif sentiment_normalized > -0.5:
                    sent_desc = "BEARISH"
                else:
                    sent_desc = "STRONGLY BEARISH"
                lines.append(f"  • FinLLM Sentiment (12% weight): {sent_desc} - Score: {sentiment_normalized:+.2f}")
                
                # Volatility (8% weight)
                if volatility > 0.03:
                    vol_risk = "HIGH VOLATILITY - High risk of sharp moves"
                elif volatility > 0.02:
                    vol_risk = "MODERATE VOLATILITY - Normal price swings"
                else:
                    vol_risk = "LOW VOLATILITY - Stable conditions"
                lines.append(f"  • Price Volatility (8% weight): {vol_risk} ({volatility:.4f})\n")
                
                # Signal Decision Logic
                lines.append("🎯 SIGNAL DECISION LOGIC:")
                
                if signal == "BUY":
                    lines.append(f"  ✅ BUY: Composite bullish signal (score: {total_signal:.3f})")
                    lines.append(f"     - Technical momentum is positive ({price_change_pct:+.2f}%)")
                    lines.append(f"     - Moving average trend is {'favorable (uptrend)' if ma_ratio > 0 else 'unfavorable (downtrend)'}")
                    lines.append(f"     - RSI indicates {'oversold conditions, potential bounce' if rsi < 40 else 'neutral-bullish conditions'}")
                    lines.append(f"     - FinLLM sentiment is bullish, supporting upside bias")
                    if volume_spike < 0.8:
                        lines.append(f"     - ⚠️  Low volume detected - execute with caution")
                elif signal == "SELL":
                    lines.append(f"  ❌ SELL: Composite bearish signal (score: {total_signal:.3f})")
                    lines.append(f"     - Technical momentum is negative ({price_change_pct:+.2f}%)")
                    lines.append(f"     - Moving average trend is {'unfavorable (downtrend)' if ma_ratio < 0 else 'challenging (uptrend)'}")
                    lines.append(f"     - RSI indicates {'overbought conditions, potential squeeze' if rsi > 60 else 'neutral-bearish conditions'}")
                    lines.append(f"     - FinLLM sentiment is bearish, supporting downside bias")
                    if volume_spike < 0.8:
                        lines.append(f"     - ⚠️  Low volume detected - execute with caution")
                else:  # HOLD
                    lines.append(f"  ⏸️  HOLD: Mixed/inconclusive signals (score: {total_signal:.3f})")
                    lines.append(f"     - Momentum is flat or indecisive ({price_change_pct:+.2f}%)")
                    lines.append(f"     - Technical indicators show conflicting messages")
                    lines.append(f"     - RSI is in neutral zone ({rsi:.1f})")
                    lines.append(f"     - Sentiment and technicals are not aligned")
                    if max_prob < 0.40:
                        lines.append(f"     - Low confidence ({round(max_prob*100, 1)}%) - wait for clearer signals")
                    else:
                        lines.append(f"     - Moderate confidence ({round(max_prob*100, 1)}%) - monitor for breakout")
                
                # Risk Factors
                lines.append("\n⚠️  RISK FACTORS:")
                risk_count = 0
                if volatility > 0.03:
                    lines.append(f"  • High volatility may increase slippage costs")
                    risk_count += 1
                if volume_spike < 0.8:
                    lines.append(f"  • Low volume reduces liquidity and signal reliability")
                    risk_count += 1
                if abs(ma_ratio) < 0.05:
                    lines.append(f"  • Moving averages converging - potential trend reversal")
                    risk_count += 1
                if (rsi > 65 and signal == "BUY") or (rsi < 35 and signal == "SELL"):
                    lines.append(f"  • Extreme RSI values may indicate overextension")
                    risk_count += 1
                
                if risk_count == 0:
                    lines.append("  • No major risk factors detected")
                
                # Recommendation
                lines.append(f"\n💡 RECOMMENDATION: {signal} ({round(confidence*100, 1)}% confidence)")
                if signal == "BUY":
                    lines.append(f"   Set entry point at current price or on dips. Target: +5-10% upside potential.")
                    lines.append(f"   Stop loss: {price_change_pct-2:.2f}% to protect against reversals.")
                elif signal == "SELL":
                    lines.append(f"   Consider taking profits or shorting with caution. Target: -5-10% downside.")
                    lines.append(f"   Stop loss: {price_change_pct+2:.2f}% to limit losses.")
                else:
                    lines.append(f"   Wait for clearer signals or accumulate at support levels.")
                    lines.append(f"   Monitor RSI, moving averages, and volume for patterns.")
                
                return "\n".join(lines)
            
            # Generate the detailed reason
            reason = generate_detailed_reason(
                ticker, signal, total_signal, feature_scores,
                price_change_pct, rsi, ma_ratio, volume_spike,
                sentiment_score, volatility, max_prob, buy_prob,
                sell_prob, hold_prob, confidence
            )
            
            # ── Build Factor Explanation ────────────────────────────────
            factor_impacts = []
            for fname, (score, weight) in feature_scores.items():
                impact = abs(score)
                direction = score > 0
                factor_impacts.append((fname, impact, direction))
            
            factor_impacts.sort(key=lambda x: x[1], reverse=True)
            
            top_factors = []
            for fname, impact, direction in factor_impacts[:4]:
                top_factors.append({
                    "feature": fname,
                    "impact": round(float(impact), 3),
                    "direction": "positive" if direction else "negative",
                })
            
            signal_doc = {
                "ticker": ticker,
                "signal": signal,
                "confidence": round(float(confidence), 2),
                "probabilities": {
                    "BUY": round(float(buy_prob), 2),
                    "HOLD": round(float(hold_prob), 2),
                    "SELL": round(float(sell_prob), 2),
                },
                "reason": reason,
                "top_factors": top_factors,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            return signal_doc
            
        except Exception as exc:
            logger.error("predict_ticker_error", ticker=ticker, error=str(exc))
            return {
                "ticker": ticker,
                "signal": "HOLD",
                "confidence": 0.5,
                "probabilities": {"BUY": 0.33, "HOLD": 0.34, "SELL": 0.33},
                "reason": f"Error in signal calculation: {str(exc)[:60]}",
                "top_factors": [],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

    def predict_all(self, tickers: list[str] | None = None) -> list[dict]:
        """Generate signals for all tracked tickers and publish to Kafka."""
        tickers = tickers or WATCHLIST
        signals = []
        for ticker in tickers:
            signal = self.predict_ticker(ticker)
            signals.append(signal)
            self.producer.publish(
                topic=self.settings.kafka_topic_signals,
                key=ticker,
                value=signal,
            )
            logger.info(
                "signal_generated",
                ticker=ticker,
                signal=signal["signal"],
                confidence=signal["confidence"],
            )
        return signals
