from __future__ import annotations

"""
FinLLaMA inference service.
Loads the model once and exposes async-friendly inference methods.
Uses 4-bit quantisation (bitsandbytes) to fit on a single GPU.
"""
import hashlib
import json
import re
import threading
from typing import Any

import redis
import structlog
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from config import get_settings
from .prompts import (
    build_sentiment_prompt,
    build_ticker_prompt,
    build_event_prompt,
    build_batch_sentiment_prompt,
)

logger = structlog.get_logger(__name__)

_LOAD_LOCK = threading.Lock()
_CACHE_TTL_SEC = 3600   # LLM outputs cached for 1 hour
_CACHE_PREFIX  = "llm_cache:"


class FinLLaMAService:
    """
    Singleton-style wrapper around FinLLaMA.
    Call FinLLaMAService.get_instance() for shared access.
    """

    _instance: "FinLLaMAService | None" = None

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model = None
        self.tokenizer = None
        self._pipe = None
        self._loaded = False
        # Redis cache — gracefully disabled if Redis is unavailable
        self._cache: redis.Redis | None = None
        try:
            self._cache = redis.from_url(self.settings.redis_url, decode_responses=True)
            self._cache.ping()
            logger.info("llm_cache_connected")
        except Exception:
            logger.warning("llm_cache_unavailable_caching_disabled")
            self._cache = None

    @classmethod
    def get_instance(cls) -> "FinLLaMAService":
        if cls._instance is None:
            with _LOAD_LOCK:
                if cls._instance is None:
                    svc = cls()
                    svc.load()
                    cls._instance = svc
        return cls._instance

    # ── Model loading ───────────────────────────────────────────────────────
    def load(self) -> None:
        model_id = self.settings.finllama_model_id
        device = self.settings.finllama_device
        logger.info("finllama_loading", model_id=model_id, device=device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }

        if self.settings.finllama_quantize and device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = device if device == "cpu" else "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self._pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.settings.finllama_max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self._loaded = True
        logger.info("finllama_loaded")

    # ── Cache helpers ───────────────────────────────────────────────────────
    def _cache_key(self, task: str, text: str) -> str:
        digest = hashlib.sha256(text.encode()).hexdigest()[:24]
        return f"{_CACHE_PREFIX}{task}:{digest}"

    def _cache_get(self, task: str, text: str) -> dict | list | None:
        if self._cache is None:
            return None
        try:
            raw = self._cache.get(self._cache_key(task, text))
            if raw:
                logger.debug("llm_cache_hit", task=task)
                return json.loads(raw)
        except Exception:
            pass
        return None

    def _cache_set(self, task: str, text: str, value: dict | list) -> None:
        if self._cache is None:
            return
        try:
            self._cache.setex(
                self._cache_key(task, text),
                _CACHE_TTL_SEC,
                json.dumps(value),
            )
        except Exception:
            pass

    # ── Low-level generation ────────────────────────────────────────────────
    def _generate(self, prompt: str) -> str:
        if not self._loaded:
            raise RuntimeError("FinLLaMA model not loaded. Call load() first.")
        outputs = self._pipe(prompt, return_full_text=False)
        return outputs[0]["generated_text"].strip()

    def _extract_json(self, raw: str) -> dict | list:
        """Extract first JSON object or array from generation output."""
        # Try to find JSON block in the output
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Fallback: try parsing entire string
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("json_parse_failed", raw=raw[:200])
            return {}

    # ── Public inference methods ────────────────────────────────────────────
    def analyse_sentiment(self, text: str) -> dict:
        """
        Returns structured sentiment for a single piece of financial text.
        Results are cached in Redis for _CACHE_TTL_SEC seconds.

        Example output:
            {
                "ticker": "TSLA",
                "sentiment": "bullish",
                "confidence": 0.82,
                "impact": "short_term_positive",
                "reason": "positive delivery expectations"
            }
        """
        cached = self._cache_get("sentiment", text)
        if cached:
            return cached

        prompt = build_sentiment_prompt(text)
        raw = self._generate(prompt)
        result = self._extract_json(raw)
        output = {
            "ticker": result.get("ticker"),
            "sentiment": result.get("sentiment", "neutral"),
            "confidence": float(result.get("confidence", 0.5)),
            "impact": result.get("impact", "neutral"),
            "reason": result.get("reason", ""),
            "raw_text": text[:200],
        }
        self._cache_set("sentiment", text, output)
        return output

    def extract_tickers(self, text: str) -> list[str]:
        """Extract stock ticker symbols from text."""
        cached = self._cache_get("tickers", text)
        if cached is not None:
            return cached

        prompt = build_ticker_prompt(text)
        raw = self._generate(prompt)
        tickers = self._extract_json(raw)
        output = [str(t).upper() for t in tickers if t] if isinstance(tickers, list) else []
        self._cache_set("tickers", text, output)
        return output

    def classify_event(self, text: str) -> dict:
        """Classify the market event type in text."""
        cached = self._cache_get("event", text)
        if cached:
            return cached

        prompt = build_event_prompt(text)
        raw = self._generate(prompt)
        result = self._extract_json(raw)
        output = {
            "event_type": result.get("event_type", "general_news"),
            "urgency": result.get("urgency", "low"),
            "affected_sectors": result.get("affected_sectors", []),
        }
        self._cache_set("event", text, output)
        return output

    def analyse_batch(self, texts: list[str]) -> list[dict]:
        """Analyse sentiment for a batch of texts (more efficient)."""
        results = []
        # Process in sub-batches to respect token limits
        batch_size = self.settings.finllama_batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            prompt = build_batch_sentiment_prompt(batch)
            raw = self._generate(prompt)
            parsed = self._extract_json(raw)
            if isinstance(parsed, list):
                results.extend(parsed)
            else:
                # Fallback: analyse individually
                for text in batch:
                    results.append(self.analyse_sentiment(text))
        return results
