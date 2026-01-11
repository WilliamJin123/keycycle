
import logging
from typing import Any

# Mock Logger
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("test_logger")

# Mock Agno Model (Pydantic-like)
class MockModel:
    def __init__(self, id: str = "default", model_id: str = None, **kwargs):
        self.id = id
        # Simulate Pydantic overwriting the attribute if it's a field
        self.model_id = model_id 
        self.kwargs = kwargs
        # Mimic Pydantic setting attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def invoke(self, *args, **kwargs):
        raise ValueError("429 Rate Limit Exceeded")

# Mixin from the file (simplified for repro)
class RotatingCredentialsMixin:
    def __init__(
        self, 
        *args, 
        model_id: str,
        wrapper=None, 
        rotating_wait=True, 
        rotating_timeout=10.0, 
        rotating_estimated_tokens=1000,
        rotating_max_retries=5, 
        logger = None,
        **kwargs):
        
        self.logger = logger or logging.getLogger(__name__)
        self.wrapper = wrapper
        
        # Call next in MRO
        super().__init__(*args, **kwargs)
        
        self.model_id = model_id

    def _is_rate_limit_error(self, e: Exception) -> bool:
        return "429" in str(e)

    def invoke(self, *args, **kwargs):
        limit = 2
        for attempt in range(limit + 1):
            try:  
                # Mock usage record
                # response = super().invoke(*args, **kwargs)
                # return response
                # Simulate call that fails
                super().invoke(*args, **kwargs)
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < limit:
                    self.logger.warning("429 Hit on key %s (Sync) [%s]. Rotating and retrying (%d/%d).", "TESTKEY", self.model_id, attempt + 1, limit)
                    continue
                # raise e

# Dynamic class creation
RotatingProviderClass = type(
    "RotatingMockModel",
    (RotatingCredentialsMixin, MockModel),
    {}
)

# Instantiate
model_id = "gemini-pro"
instance = RotatingProviderClass(
    model_id=model_id,
    id=model_id,
    logger=logger
)

print(f"Instance model_id: {instance.model_id}")

# Trigger invoke
try:
    instance.invoke("test")
except Exception:
    pass
