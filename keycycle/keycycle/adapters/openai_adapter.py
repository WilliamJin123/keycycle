import logging
import time
from typing import List, Any, Optional, Union, Callable, Generator, AsyncGenerator

from ..config.dataclasses import KeyUsage, RateLimits
from ..key_rotation.rotation_manager import RotatingKeyManager

logger = logging.getLogger(__name__)

try:
    import openai
    from openai import OpenAI, AsyncOpenAI, RateLimitError, APIError
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    # Generic placeholders
    OpenAI = object
    AsyncOpenAI = object
    RateLimitError = Exception
    APIError = Exception

PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "cerebras": "https://api.cerebras.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "cohere": "https://api.cohere.ai/compatibility/v1",

}

class BaseRotatingClient:
    def __init__(self, 
        manager: RotatingKeyManager, 
        limit_resolver: Callable[[str], RateLimits],
        default_model: str, 
        estimated_tokens: int = 1000, 
        max_retries: int = 5,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        client_kwargs: dict = None
    ):
        """
        Initialize the rotating client.
        
        Args:
            manager: The key rotation manager
            limit_resolver: Function to resolve rate limits for a model
            default_model: Default model to use
            estimated_tokens: Estimated tokens per request
            max_retries: Maximum number of retries on rate limit
            base_url: Base URL for the API (takes precedence over provider)
            provider: Provider name (openai, openrouter, gemini, cerebras, groq)
            client_kwargs: Additional kwargs to pass to OpenAI client
        """
        
        if not HAS_OPENAI:
            raise ImportError("The 'openai' library is required. Install with `pip install openai`.")
            
        self.manager = manager
        self.limit_resolver = limit_resolver
        self.default_model = default_model
        self.estimated_tokens = estimated_tokens
        self.max_retries = max_retries
        self.client_kwargs = client_kwargs or {}

        if base_url:
            self.base_url = base_url
        elif provider:
            provider_lower = provider.lower()
            if provider_lower not in PROVIDER_BASE_URLS:
                raise ValueError(
                    f"Unknown provider: {provider}. "
                    f"Valid providers: {', '.join(PROVIDER_BASE_URLS.keys())}"
                )
            self.base_url = PROVIDER_BASE_URLS[provider_lower]
        else:
            self.base_url = None  # Will use OpenAI's default

        if self.base_url:
            self.client_kwargs['base_url'] = self.base_url

    def _is_rate_limit_error(self, e: Exception) -> bool:
        if isinstance(e, RateLimitError):
            return True
        err_str = str(e).lower()
        flagged_strs = ["429", "too many requests", "rate limit", "resource exhausted", "traffic", "rate-limited"]
        if any(i in err_str for i in 
                   flagged_strs):
            return True

        if hasattr(e, "status_code") and e.status_code == 429:
            return True
        body = getattr(e, "body", None) or getattr(e, "response", None)
        if body:
            body_str = str(body).lower()
            if any(i in body_str for i in flagged_strs):
                return True
        return False

    def _record_usage(self, key_usage: KeyUsage, model_id: str, actual_tokens: int):
        self.manager.record_usage(
            key_obj=key_usage, 
            model_id=model_id, 
            actual_tokens=actual_tokens, 
            estimated_tokens=self.estimated_tokens
        )
        
    def _extract_usage(self, response: Any) -> int:
        try:
            if hasattr(response, 'usage') and response.usage:
                return response.usage.total_tokens
        except:
            pass
        return 0

# --- SYNC IMPLEMENTATION ---

class RotatingOpenAIClient(BaseRotatingClient):
    def _get_fresh_client(self, api_key: str):
        return OpenAI(api_key=api_key, **self.client_kwargs)

    def __getattr__(self, name):
        return SyncProxyHelper(self, [name])

    def _execute(self, path: List[str], args, kwargs):
        model_id = kwargs.get('model', self.default_model)
        if 'model' not in kwargs:
            kwargs['model'] = model_id
        limits = self.limit_resolver(model_id)
        
        for attempt in range(self.max_retries + 1):
            key_usage = self.manager.get_key(model_id, limits, self.estimated_tokens)
            if not key_usage:
                raise RuntimeError(f"No available keys for {model_id}")

            try:
                real_client = self._get_fresh_client(key_usage.api_key)
                
                target = real_client
                for p in path:
                    target = getattr(target, p)
                
                result = target(*args, **kwargs)
                
                if kwargs.get('stream', False):
                    return self._wrap_stream(result, key_usage, model_id)
                
                self._record_usage(key_usage, model_id, self._extract_usage(result))
                return result

            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < self.max_retries:
                    logger.warning(f"429/RateLimit hit for {model_id} on key ...{key_usage.api_key[-8:]}. Rotating. (Attempt {attempt + 1}/{self.max_retries + 1})")
                    key_usage.trigger_cooldown()
                    self.manager.force_rotate_index()
                    time.sleep(0.5)
                    continue
                self._record_usage(key_usage, model_id, 0) 
                raise e

    def _wrap_stream(self, generator: Generator, key_usage: KeyUsage, model_id: str):
        accumulated_tokens = 0
        try:
            for chunk in generator:
                if hasattr(chunk, 'usage') and chunk.usage:
                    accumulated_tokens = chunk.usage.total_tokens
                yield chunk
        except Exception as e:
            if self._is_rate_limit_error(e):
                logger.warning(f"Rate limit hit during streaming for {model_id} on key ...{key_usage.api_key[-8:]}.")
                key_usage.trigger_cooldown()
                self.manager.force_rotate_index()
            raise
        finally:
            self._record_usage(key_usage, model_id, accumulated_tokens)

class SyncProxyHelper:
    def __init__(self, client: RotatingOpenAIClient, path: List[str]):
        self.client = client
        self.path = path

    def __getattr__(self, name):
        return SyncProxyHelper(self.client, self.path + [name])

    def __call__(self, *args, **kwargs):
        return self.client._execute(self.path, args, kwargs)


# --- ASYNC IMPLEMENTATION ---

class RotatingAsyncOpenAIClient(BaseRotatingClient):
    def _get_fresh_client(self, api_key: str):
        return AsyncOpenAI(api_key=api_key, **self.client_kwargs)

    def __getattr__(self, name):
        return AsyncProxyHelper(self, [name])

    async def _execute(self, path: List[str], args, kwargs: dict):
        model_id = kwargs.get('model', self.default_model)
        if 'model' not in kwargs:
            kwargs['model'] = model_id
        limits = self.limit_resolver(model_id)
        
        if kwargs.get('stream', False) and 'stream_options' not in kwargs:
            kwargs['stream_options'] = {"include_usage": True}
        
        if kwargs.get('stream', False) and 'stream_options' not in kwargs:
            kwargs['stream_options'] = {"include_usage": True}

        for attempt in range(self.max_retries + 1):
            key_usage = self.manager.get_key(model_id, limits, self.estimated_tokens)
            if not key_usage:
                raise RuntimeError(f"No available keys for {model_id}")

            try:
                real_client = self._get_fresh_client(key_usage.api_key)
                
                target = real_client
                for p in path:
                    target = getattr(target, p)
                
                result = await target(*args, **kwargs)
                
                if kwargs.get('stream', False):
                    return self._wrap_stream(result, key_usage, model_id)
                
                self._record_usage(key_usage, model_id, self._extract_usage(result))
                return result

            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < self.max_retries:
                    logger.warning(f"429/RateLimit hit for {model_id} on key ...{key_usage.api_key[-8:]}. Rotating. (Attempt {attempt + 1}/{self.max_retries + 1})")
                    key_usage.trigger_cooldown()
                    self.manager.force_rotate_index()
                    time.sleep(0.5)
                    continue
                self._record_usage(key_usage, model_id, 0) 
                raise e

    async def _wrap_stream(self, generator: AsyncGenerator, key_usage: KeyUsage, model_id: str):
        accumulated_tokens = 0
        try:
            async for chunk in generator:
                if hasattr(chunk, 'usage') and chunk.usage:
                    accumulated_tokens = chunk.usage.total_tokens
                yield chunk
        except Exception as e:
            if self._is_rate_limit_error(e):
                logger.warning(f"Rate limit hit during streaming for {model_id} on key ...{key_usage.api_key[-8:]}.")
                key_usage.trigger_cooldown()
                self.manager.force_rotate_index()
            raise
        finally:
            self._record_usage(key_usage, model_id, accumulated_tokens)

class AsyncProxyHelper:
    def __init__(self, client: RotatingAsyncOpenAIClient, path: List[str]):
        self.client = client
        self.path = path

    def __getattr__(self, name):
        return AsyncProxyHelper(self.client, self.path + [name])

    async def __call__(self, *args, **kwargs):
        return await self.client._execute(self.path, args, kwargs)
