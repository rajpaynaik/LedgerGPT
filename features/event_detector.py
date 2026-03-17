from __future__ import annotations

"""
Event type detection using FinLLM classification.
Identifies event types (earnings, product, macro, rumor, general) for sentiment context.
"""

from enum import Enum
from typing import Optional
import json
import structlog

logger = structlog.get_logger(__name__)


class EventType(str, Enum):
    """Event classification types."""
    EARNINGS = "earnings"
    PRODUCT = "product"
    MACRO = "macro"
    RUMOR = "rumor"
    GENERAL = "general"


class EventImpact(str, Enum):
    """Event impact duration classification."""
    SHORT_TERM = "short_term"      # < 1 day impact
    MEDIUM_TERM = "medium_term"    # 1-7 days impact
    LONG_TERM = "long_term"        # > 7 days impact


class EventDetector:
    """
    Detects event types and impact from financial text using pattern matching and keywords.
    Can be enhanced with FinLLM classification endpoint.
    """
    
    # Event keyword patterns
    KEYWORDS = {
        EventType.EARNINGS: {
            "keywords": ["earnings", "eps", "revenue", "guidance", "beat estimates", 
                        "miss estimates", "quarterly results", "fy guidance"],
            "confidence": 0.95
        },
        EventType.PRODUCT: {
            "keywords": ["launch", "product", "feature", "release", "announced",
                        "new model", "unveil", "introduce", "upcoming"],
            "confidence": 0.85
        },
        EventType.MACRO: {
            "keywords": ["fed", "federal reserve", "interest rate", "inflation",
                        "gdp", "unemployment", "cpi", "sector rotation",
                        "economic data", "policy"],
            "confidence": 0.80
        },
        EventType.RUMOR: {
            "keywords": ["rumor", "specul", "report", "allegedly", "sources say",
                        "unconfirmed", "may", "could", "expected to"],
            "confidence": 0.70
        }
    }
    
    @classmethod
    def detect_event_type(cls, text: str) -> dict:
        """
        Detect event type from financial text using keyword matching.
        
        Args:
            text: Financial text to analyze
            
        Returns:
            {
                'event_type': EventType,
                'confidence': float (0-1),
                'keywords_matched': list[str],
                'impact': EventImpact
            }
        """
        text_lower = text.lower()
        best_match = None
        max_confidence = 0
        matched_keywords = []
        
        # Check each event type's keywords
        for event_type, pattern_info in cls.KEYWORDS.items():
            keywords = pattern_info['keywords']
            base_confidence = pattern_info['confidence']
            
            # Count keyword matches
            matches = [kw for kw in keywords if kw in text_lower]
            
            if matches:
                # Confidence increases with more matches
                match_count = len(matches)
                confidence = min(0.99, base_confidence * (1 + 0.1 * (match_count - 1)))
                
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_match = event_type
                    matched_keywords = matches
        
        # Default to GENERAL if no specific match
        if not best_match:
            best_match = EventType.GENERAL
            max_confidence = 0.5
        
        # Infer impact duration
        impact = cls._infer_impact(best_match, text)
        
        return {
            'event_type': best_match.value,
            'confidence': round(max_confidence, 3),
            'keywords_matched': matched_keywords,
            'impact': impact
        }
    
    @classmethod
    def _infer_impact(cls, event_type: EventType, text: str) -> str:
        """
        Infer event impact duration.
        
        Args:
            event_type: Detected event type
            text: Financial text
            
        Returns:
            EventImpact string
        """
        # Earnings typically have long-term impact
        if event_type == EventType.EARNINGS:
            return EventImpact.LONG_TERM.value
        
        # Product launches have medium to long-term impact
        elif event_type == EventType.PRODUCT:
            if any(word in text.lower() for word in ["major", "flagship", "breakthrough", "revolutionary"]):
                return EventImpact.LONG_TERM.value
            return EventImpact.MEDIUM_TERM.value
        
        # Macro events have medium to long-term impact
        elif event_type == EventType.MACRO:
            if any(word in text.lower() for word in ["fed", "policy", "rate"]):
                return EventImpact.LONG_TERM.value
            return EventImpact.MEDIUM_TERM.value
        
        # Rumors have short to medium-term impact
        elif event_type == EventType.RUMOR:
            return EventImpact.SHORT_TERM.value
        
        # General events default to medium-term
        else:
            return EventImpact.MEDIUM_TERM.value
    
    @classmethod
    def get_event_impact_weight(cls, impact: str) -> float:
        """
        Get weight multiplier for signal confidence based on event impact.
        
        Args:
            impact: EventImpact string
            
        Returns:
            Weight multiplier (0.8-1.2)
        """
        weights = {
            EventImpact.SHORT_TERM.value: 0.8,      # Less reliable
            EventImpact.MEDIUM_TERM.value: 1.0,     # Normal weight
            EventImpact.LONG_TERM.value: 1.2        # More reliable/persistent
        }
        return weights.get(impact, 1.0)
    
    @classmethod
    def format_event_context(cls, detection: dict) -> str:
        """
        Format event detection result for signal reasoning.
        
        Args:
            detection: Event detection result dict
            
        Returns:
            Formatted string for display
        """
        event_type = detection['event_type'].upper()
        confidence = detection['confidence']
        impact = detection['impact'].replace('_', ' ').upper()
        keywords = ', '.join(detection['keywords_matched'][:3])
        
        emoji_map = {
            'EARNINGS': '📊',
            'PRODUCT': '🚀',
            'MACRO': '📈',
            'RUMOR': '⚠️',
            'GENERAL': '📰'
        }
        emoji = emoji_map.get(event_type, '📋')
        
        if keywords:
            return f"{emoji} {event_type} ({confidence*100:.0f}% confidence) - {impact} impact\n   Keywords: {keywords}"
        else:
            return f"{emoji} {event_type} ({confidence*100:.0f}% confidence) - {impact} impact"


# Example FinLLM integration (requires FinLLM endpoint enhancement)
class FinLLMEventDetector:
    """
    Advanced event detection using FinLLM classification.
    Requires FinLLM endpoint: POST /event-classify with text input.
    Falls back to keyword matching if unavailable.
    """
    
    def __init__(self, finllm_url: str = "http://localhost:8000"):
        """
        Initialize FinLLM event detector.
        
        Args:
            finllm_url: Base URL for FinLLM service
        """
        self.finllm_url = finllm_url
        self.fallback = EventDetector()
    
    def detect(self, text: str) -> dict:
        """
        Detect event type using FinLLM (with fallback).
        
        Args:
            text: Financial text
            
        Returns:
            Event detection result
        """
        # Try FinLLM first (would require endpoint extension)
        # For now, use fallback
        return self.fallback.detect_event_type(text)
