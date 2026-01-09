import functools
import logging
from typing import AsyncIterator, Iterator, Optional
from ..config.dataclasses import KeyUsage
from ..config.log_config import default_logger
from agno.models.response import ModelResponse

class RotatingCredentialsMixin:
    """
    Mixin that handles key rotation, 429 detection, and 30s cooldown triggers.
    """
    
    def __init__(
        self, 
        *args, 
        wrapper=None, 
        rotating_wait=True, 
        rotating_timeout=10.0, 
        rotating_estimated_tokens=1000,
        rotating_max_retries=5, 
        logger = None,
        **kwargs):
        """
        Initializes rotation parameters and patches the model if necessary.
        
        Args:
            wrapper: The MultiProviderWrapper instance managing keys.
            rotating_wait: Whether to wait for a key if none are available.
            rotating_timeout: Max time to wait for a key.
            rotating_estimated_tokens: Default token estimate for rate limiting.
            rotating_max_retries: Number of retries on 429 errors.
        """
        self.logger = logger or default_logger
        self.wrapper = wrapper
        self._rotating_wait = rotating_wait
        self._rotating_timeout = rotating_timeout
        self._estimated_tokens = rotating_estimated_tokens
        self._max_retries = rotating_max_retries

        super().__init__(*args, **kwargs)

    def __init_subclass__(cls, **kwargs):
        """
        Hook to automatically copy function signatures and docstrings 
        from the parent class to this mixin's wrapper methods.
        """
        super().__init_subclass__(**kwargs)
        methods_to_sync = ['invoke', 'ainvoke', 'invoke_stream', 'ainvoke_stream']
        for method_name in methods_to_sync:
            if not hasattr(cls, method_name):
                continue
            target_method = None
            for parent in cls.mro():
                if parent is cls or parent is RotatingCredentialsMixin:
                    continue
                # Check if the method is defined in this specific parent's __dict__
                if hasattr(parent, method_name) and method_name in parent.__dict__:
                    target_method = getattr(parent, method_name)
                    break
            
            if target_method:
                mixin_implementation = getattr(cls, method_name)
                wrapped_method = functools.wraps(target_method)(mixin_implementation)

                if hasattr(target_method, '__signature__'):
                    wrapped_method.__signature__ = target_method.__signature__
                setattr(cls, method_name, wrapped_method)

    def _rotate_credentials(self) -> KeyUsage:
        key_usage: KeyUsage = self.wrapper.get_key_usage(
            model_id=self.id,
            estimated_tokens=self._estimated_tokens,
            wait=self._rotating_wait,
            timeout=self._rotating_timeout
        )
        self.api_key = key_usage.api_key
        
        if hasattr(self, "client"): self.client = None
        if hasattr(self, "async_client"): self.async_client = None
        if hasattr(self, "gemini_client"): self.gemini_client = None

        return key_usage

    def _is_rate_limit_error(self, e: Exception) -> bool:
        """Heuristic to detect rate limits across different providers"""
        err_str = str(e).lower()
        return any(indicator in err_str for indicator in 
                   ["429", "too many requests", "rate limit", "resource exhausted"])

    def _get_retry_limit(self):
        return min(self._max_retries, len(self.wrapper.manager.keys) - 1)

    def _record_usage(self, key_obj: KeyUsage, response: Optional[ModelResponse]):
        """
        Extracts usage from the response and reports it to the manager.
        Falls back to estimated_tokens if response is None or usage is missing.
        """
        if not self.wrapper:
            return

        actual_tokens = 0
        
        if response:
            actual_tokens = response.response_usage.total_tokens

        # Fallback: If no usage found (or streaming), use the estimate
        if actual_tokens == 0:
            actual_tokens = self._estimated_tokens

        self.wrapper.manager.record_usage(
            key_obj=key_obj,
            model_id=self.id,
            actual_tokens=actual_tokens,
            estimated_tokens=self._estimated_tokens
        )


    def invoke(self, *args, **kwargs) -> ModelResponse:
        limit = self._get_retry_limit()
        
        for attempt in range(limit + 1):
            key_usage = self._rotate_credentials()
            try:  
                response = super().invoke(*args, **kwargs)
                self._record_usage(key_usage, response)
                return response
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < limit:
                    self.logger.warning("429 Hit on key %s (Sync). Rotating and retrying (%d/%d).", self.api_key[-8:], attempt + 1, limit)
                    key_usage.trigger_cooldown()
                    self.wrapper.manager.force_rotate_index()
                    continue
                raise e

    async def ainvoke(self, *args, **kwargs) -> ModelResponse:
        limit = self._get_retry_limit()
        
        for attempt in range(limit + 1):
            key_usage = self._rotate_credentials()
            try:
                response = await super().ainvoke(*args, **kwargs)
                self._record_usage(key_usage, response)
                return response
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < limit:
                    # print(f" 429 Hit on key ...{self.api_key[-8:]} (Async). Rotating and retrying ({attempt+1}/{limit})...")
                    self.logger.warning("429 Hit on key %s (Async). Rotating and retrying (%d/%d).", self.api_key[-8:], attempt + 1, limit)
                    key_usage.trigger_cooldown()
                    self.wrapper.manager.force_rotate_index()
                    continue
                raise e
    
    def invoke_stream(self, *args, **kwargs) -> Iterator[ModelResponse]:
        limit = self._get_retry_limit()
        
        for attempt in range(limit + 1):
            key_usage = self._rotate_credentials()
            try:  
                stream = super().invoke_stream(*args, **kwargs)
                final_usage = None
                for chunk in stream:
                    if chunk.response_usage:
                        final_usage = chunk.response_usage
                    yield chunk
                if final_usage:
                    dummy_response = ModelResponse()
                    dummy_response.response_usage = final_usage
                    self._record_usage(key_usage, dummy_response)
                else:
                    # If the stream didn't return usage data, fallback to estimation
                    self._record_usage(key_usage, None)
                    
                return
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < limit:
                    # print(f" 429 Hit on key ...{self.api_key[-8:]} (Sync Stream). Rotating and retrying ({attempt+1}/{limit})...")
                    self.logger.warning("429 Hit on key %s (Sync Stream). Rotating and retrying (%d/%d).", self.api_key[-8:], attempt + 1, limit)
                    key_usage.trigger_cooldown()
                    self.wrapper.manager.force_rotate_index()
                    continue
                raise e

    async def ainvoke_stream(self, *args, **kwargs) -> AsyncIterator[ModelResponse]:
        limit = self._get_retry_limit()
        
        for attempt in range(limit + 1):
            key_usage = self._rotate_credentials()
            try:
                stream = super().ainvoke_stream(*args, **kwargs)
                
                final_usage = None
                async for chunk in stream:
                    if chunk.response_usage:
                        final_usage = chunk.response_usage
                    yield chunk
                
                # Stream completed successfully. Record usage.
                if final_usage:
                    dummy_response = ModelResponse()
                    dummy_response.response_usage = final_usage
                    self._record_usage(key_usage, dummy_response)
                else:
                    self._record_usage(key_usage, None)
                
                return
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < limit:
                    # print(f" 429 Hit on key ...{self.api_key[-8:]} (Async Stream). Rotating and retrying ({attempt+1}/{limit})...")
                    self.logger.warning("429 Hit on key %s (Async Stream). Rotating and retrying (%d/%d).", self.api_key[-8:], attempt + 1, limit)
                    key_usage.trigger_cooldown()
                    self.wrapper.manager.force_rotate_index()
                    continue
                raise e
