"""
Standalone Sentiment Worker Script
Runs the FinLLM sentiment analysis pipeline.

Usage:
    python -m scripts.run_sentiment_worker --mode sample
    python -m scripts.run_sentiment_worker --mode interactive
    python -m scripts.run_sentiment_worker --mode monitor --tickers AAPL,TSLA
"""

import argparse
import time
from datetime import datetime
import structlog

from config import get_settings
from llm.sentiment_worker import SentimentWorker
from database.sentiment_crud import SentimentDB
from features.ticker_detection import VALID_TICKERS

logger = structlog.get_logger(__name__)


def run_continuous_monitoring(worker: SentimentWorker, tickers: list[str], interval: int = 300):
    """
    Run sentiment monitoring in continuous mode.
    
    Args:
        worker: SentimentWorker instance
        tickers: List of tickers to monitor
        interval: Polling interval in seconds (default 5 minutes)
    """
    logger.info("Starting continuous sentiment monitoring", tickers=tickers, interval=interval)
    
    try:
        while True:
            for ticker in tickers:
                try:
                    # Get latest sentiment
                    sentiment = worker.get_latest_sentiment(ticker)
                    summary = worker.get_sentiment_summary(ticker)
                    
                    if sentiment:
                        logger.info(
                            "Sentiment retrieved",
                            ticker=ticker,
                            score=sentiment['sentiment_score'],
                            confidence=sentiment['confidence']
                        )
                    
                    if summary:
                        logger.info(
                            "Sentiment summary retrieved",
                            ticker=ticker,
                            mentions=summary['mention_count'],
                            avg_sentiment=summary['avg_sentiment'],
                            bullish=summary['bullish_count'],
                            bearish=summary['bearish_count']
                        )
                
                except Exception as e:
                    logger.error("Error retrieving sentiment", ticker=ticker, error=str(e))
            
            logger.info("Waiting for next polling cycle", interval=interval)
            time.sleep(interval)
    
    except KeyboardInterrupt:
        logger.info("Sentiment monitoring stopped by user")
    except Exception as e:
        logger.error("Sentiment monitoring failed", error=str(e))


def run_sample_analysis(worker: SentimentWorker):
    """
    Run sample sentiment analysis on predefined texts.
    
    Args:
        worker: SentimentWorker instance
    """
    logger.info("Running sample sentiment analysis")
    
    sample_texts = [
        "$AAPL released groundbreaking AI features today. Investors are very bullish on the announcement.",
        "TSLA stock drops 5% after disappointing earnings. Analysts remain concerned about supply chain issues.",
        "GOOGL launches new cloud infrastructure. Enterprise clients report high satisfaction.",
        "$NVDA continues to dominate AI chip market. Record revenue expected next quarter.",
        "Meta's metaverse division faces challenges. Facebook reported weaker user engagement.",
    ]
    
    for i, text in enumerate(sample_texts, 1):
        logger.info(f"Analyzing sample {i}/{len(sample_texts)}")
        result = worker.analyze_text(text, source="sample")
        
        logger.info(
            f"Sample {i} analysis complete",
            tickers_detected=result['tickers_detected'],
            results=result['results']
        )
        
        time.sleep(2)  # Small delay between analyses


def run_interactive_mode(worker: SentimentWorker):
    """
    Run sentiment worker in interactive mode.
    User can enter financial text and get sentiment analysis.
    
    Args:
        worker: SentimentWorker instance
    """
    logger.info("Starting interactive sentiment analysis mode")
    print("\n" + "="*60)
    print("LedgerGPT Sentiment Worker - Interactive Mode")
    print("="*60)
    print("Enter financial text to analyze (type 'quit' to exit)")
    print("Example: '$AAPL stock surges on new iPhone announcement'")
    print("="*60 + "\n")
    
    try:
        while True:
            text = input(">> ").strip()
            
            if text.lower() == 'quit':
                logger.info("Interactive mode stopped by user")
                break
            
            if len(text) < 10:
                print("Please enter at least 10 characters of financial text.")
                continue
            
            print("\nAnalyzing...")
            result = worker.analyze_text(text, source="interactive")
            
            print(f"\nTickers detected: {result.get('tickers_detected', 0)}")
            
            for ticker, details in result.get('results', {}).items():
                print(f"\n{ticker}:")
                print(f"  - Sentiment Score: {details.get('sentiment_score', 'N/A')}")
                print(f"  - Confidence: {details.get('confidence', 'N/A')}")
                print(f"  - Event Type: {details.get('event_type', 'general')}")
                print(f"  - Stored: {details.get('stored', False)}")
            
            print()
    
    except KeyboardInterrupt:
        logger.info("Interactive mode stopped")
    except Exception as e:
        logger.error("Interactive mode failed", error=str(e))


def main():
    """Main entry point for sentiment worker script."""
    parser = argparse.ArgumentParser(
        description="LedgerGPT Sentiment Worker - FinLLM Sentiment Analysis Pipeline"
    )
    
    parser.add_argument(
        "--mode",
        choices=["interactive", "sample", "monitor"],
        default="sample",
        help="Execution mode (default: sample)"
    )
    
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL,TSLA,GOOGL,NVDA,META",
        help="Comma-separated list of tickers to monitor (for monitor mode)"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Polling interval in seconds (for monitor mode, default: 300)"
    )
    
    args = parser.parse_args()
    
    logger.info("Sentiment Worker Starting", mode=args.mode)
    
    # Initialize worker
    settings = get_settings()
    db = SentimentDB()
    worker = SentimentWorker(db=db)
    
    # Run based on selected mode
    if args.mode == "interactive":
        run_interactive_mode(worker)
    
    elif args.mode == "sample":
        run_sample_analysis(worker)
    
    elif args.mode == "monitor":
        tickers = args.tickers.split(",")
        tickers = [t.strip().upper() for t in tickers]
        
        # Validate tickers
        invalid = [t for t in tickers if t not in VALID_TICKERS]
        if invalid:
            logger.warning("Some tickers not in valid list", invalid=invalid)
        
        run_continuous_monitoring(worker, tickers, args.interval)
    
    logger.info("Sentiment Worker Stopped")


if __name__ == "__main__":
    main()
