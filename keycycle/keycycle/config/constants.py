import os
from .dataclasses import RateLimits
from .enums import RateLimitStrategy
from typing import Any, TypedDict

class ModelDict(TypedDict):
    """All the provider currrently supported"""
    cerebras: Any
    groq: Any
    gemini: Any
    openrouter: Any
    cohere: Any

PROVIDER_STRATEGIES: ModelDict = {
    'cerebras': RateLimitStrategy.PER_MODEL,
    'groq': RateLimitStrategy.PER_MODEL,
    'gemini': RateLimitStrategy.PER_MODEL,
    'openrouter': RateLimitStrategy.GLOBAL,
    'cohere': RateLimitStrategy.PER_MODEL
}

COHERE_TIERS = {
    'free': RateLimits(20, 1200, 72000),
    'pro': RateLimits(500, 30000, 1440000),
    'enterprise': RateLimits(1000, 60000, 2880000),
}

MODEL_LIMITS: ModelDict = {
    'cerebras': {
        'gpt-oss-120b': RateLimits(30, 900, 14400, 60000, 1000000, 1000000),
        'llama3.1-8b': RateLimits(30, 900, 14400, 60000, 1000000, 1000000),
        'llama-3.3-70b': RateLimits(30, 900, 14400, 60000, 1000000, 1000000),
        'qwen-3-32b': RateLimits(30, 900, 14400, 60000, 1000000, 1000000),
        'qwen-3-235b-a22b-instruct-2507': RateLimits(30, 900, 14400, 60000, 1000000, 1000000),
        'zai-glm-4.6': RateLimits(10, 100, 100, 150000, 1000000, 1000000),
        'zai-glm-4.7': RateLimits(10, 100, 100, 150000, 1000000, 1000000),
    },
    'groq': {
        'allam-2-7b': RateLimits(30, 1800, 7000, 6000, 360000, 500000),
        'canopylabs/orpheus-arabic-saudi': RateLimits(10, 100, 1200, 3600),
        'canopylabs/orpheus-v1-english': RateLimits(10, 100, 1200, 3600),
        'groq/compound': RateLimits(30, 250, 250, 70000, None, None),
        'groq/compound-mini': RateLimits(30, 250, 250, 70000, None, None),
        'llama-3.1-8b-instant': RateLimits(30, 1800, 14400, 6000, 360000, 500000),
        'llama-3.3-70b-versatile': RateLimits(30, 1000, 1000, 12000, 720000, 100000),
        'meta-llama/llama-4-maverick-17b-128e-instruct': RateLimits(30, 1000, 1000, 6000, 360000, 500000),
        'meta-llama/llama-4-scout-17b-16e-instruct': RateLimits(30, 1000, 1000, 30000, 1800000, 500000),
        'meta-llama/llama-guard-4-12b': RateLimits(30, 1800, 14400, 15000, 900000, 500000),
        'meta-llama/llama-prompt-guard-2-22m': RateLimits(30, 1800, 14400, 15000, 900000, 500000),
        'meta-llama/llama-prompt-guard-2-86m': RateLimits(30, 1800, 14400, 15000, 900000, 500000),
        'moonshotai/kimi-k2-instruct': RateLimits(60, 1000, 1000, 10000, 600000, 300000),
        'moonshotai/kimi-k2-instruct-0905': RateLimits(60, 1000, 1000, 10000, 600000, 300000),
        'openai/gpt-oss-120b': RateLimits(30, 1000, 1000, 8000, 480000, 200000),
        'openai/gpt-oss-20b': RateLimits(30, 1000, 1000, 8000, 480000, 200000),
        'openai/gpt-oss-safeguard-20b': RateLimits(30, 1000, 1000, 8000, 480000, 200000),
        'playai-tts': RateLimits(10, 100, 100, 1200, 72000, 3600),
        'playai-tts-arabic': RateLimits(10, 100, 100, 1200, 72000, 3600),
        'qwen/qwen3-32b': RateLimits(60, 1000, 1000, 6000, 360000, 500000),
        'whisper-large-v3': RateLimits(20, 2000, 2000),
        'whisper-large-v3-turbo': RateLimits(20, 2000, 2000),
    },
    'gemini': {
        'gemini-2.5-flash': RateLimits(5, 300, 20, 250000, 15000000),
        'gemini-2.5-flash-lite': RateLimits(10, 600, 20, 250000, 15000000),
        'gemini-2.5-flash-tts': RateLimits(3, 180, 10, 10000, 600000),
        'gemini-robotics-er-1.5-preview': RateLimits(10, 600, 250, 250000, 15000000),
        'gemma-3-12b': RateLimits(30, 1800, 14400, 15000, 900000),
        'gemma-3-1b': RateLimits(30, 1800, 14400, 15000, 900000),
        'gemma-3-27b': RateLimits(30, 1800, 14400, 15000, 900000),
        'gemma-3-2b': RateLimits(30, 1800, 14400, 15000, 900000),
        'gemma-3-4b': RateLimits(30, 1800, 14400, 15000, 900000),
    },
    'openrouter': {
        'default': RateLimits(20, 50, 50),
    },
    'cohere':{
        # Same for every model
        'default': COHERE_TIERS.get(os.getenv('COHERE_TIER', 'free').lower(), COHERE_TIERS['free'])
    }
}
