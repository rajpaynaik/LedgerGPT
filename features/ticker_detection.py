from __future__ import annotations

"""
Detect stock tickers from financial text using pattern matching.
Handles common ticker formats: $AAPL, AAPL, TSLA, etc.
"""

import re
from typing import Set


# Common stock tickers (expand as needed)
VALID_TICKERS = {
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
    'V', 'JNJ', 'WMT', 'PG', 'MA', 'BA', 'INTC', 'IBM', 'CSCO', 'AMD',
    'NFLX', 'ADBE', 'CRM', 'INTU', 'SNPS', 'CDNS', 'ASML', 'AVGO', 'ARM',
    'QCOM', 'AVGO', 'MU', 'LRCX', 'KLAC', 'MCHP', 'SYK', 'AON', 'LMT',
    'HON', 'NOC', 'RTX', 'CAT', 'DE', 'COP', 'CVX', 'XOM', 'CL', 'GLD',
    'SLV', 'USO', 'DBC', 'EWJ', 'EWG', 'EWU', 'FXI', 'MCHI', 'SPY', 'QQQ',
    'IWM', 'DIA', 'TLT', 'IEF', 'SHY', 'GLD', 'BND', 'BIL', 'VWO', 'IEMG'
}


def detect_tickers(text: str) -> Set[str]:
    """
    Extract stock tickers from text.
    
    Patterns:
    - $AAPL (explicit cashtag)
    - AAPL (all caps, 1-5 chars, in valid list)
    - Apple Inc (AAPL) (parenthesized)
    
    Args:
        text: Financial text or social media post
        
    Returns:
        Set of detected valid stock tickers (uppercase)
    """
    if not text:
        return set()
    
    detected = set()
    
    # Pattern 1: Explicit cashtags ($AAPL)
    cashtag_pattern = r'\$([A-Z]{1,5})\b'
    cashtags = re.findall(cashtag_pattern, text)
    for ticker in cashtags:
        if ticker in VALID_TICKERS:
            detected.add(ticker)
    
    # Pattern 2: All caps word, 1-5 chars, in valid ticker list
    # (but avoid common English words)
    caps_pattern = r'\b([A-Z]{1,5})\b'
    caps_words = re.findall(caps_pattern, text)
    for ticker in caps_words:
        if ticker in VALID_TICKERS and len(ticker) >= 2:
            detected.add(ticker)
    
    # Pattern 3: Parenthesized tickers (Company Inc (AAPL))
    paren_pattern = r'\(([A-Z]{1,5})\)'
    paren_tickers = re.findall(paren_pattern, text)
    for ticker in paren_tickers:
        if ticker in VALID_TICKERS:
            detected.add(ticker)
    
    return detected


def extract_ticker_mentions(text: str) -> dict[str, int]:
    """
    Count occurrences of each ticker in text.
    
    Args:
        text: Financial text
        
    Returns:
        Dict mapping ticker to mention count
    """
    tickers = detect_tickers(text)
    mention_counts = {}
    
    for ticker in tickers:
        # Count both $TICKER and TICKER patterns
        cashtag_count = len(re.findall(rf'\${ticker}\b', text, re.IGNORECASE))
        ticker_count = len(re.findall(rf'\b{ticker}\b', text))
        mention_counts[ticker] = cashtag_count + ticker_count
    
    return mention_counts


def add_ticker_to_whitelist(ticker: str) -> None:
    """Add a new ticker to the valid tickers set."""
    if 2 <= len(ticker) <= 5 and ticker.isalpha():
        VALID_TICKERS.add(ticker.upper())


def remove_ticker_from_whitelist(ticker: str) -> None:
    """Remove a ticker from the valid list."""
    VALID_TICKERS.discard(ticker.upper())
