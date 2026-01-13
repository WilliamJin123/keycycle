import unittest
from unittest.mock import MagicMock
import sys
import os

# Add project root to sys.path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from keycycle.adapters.openai_adapter import BaseRotatingClient
    from keycycle.key_rotation.rotating_mixin import RotatingCredentialsMixin
except ImportError:
    # Fallback for different path structures or if run directly
    from keycycle.keycycle.adapters.openai_adapter import BaseRotatingClient
    from keycycle.keycycle.key_rotation.rotating_mixin import RotatingCredentialsMixin

class MockAPIError(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        # Some libraries put body in response
        self.response = body

class TestRateLimitLogic(unittest.TestCase):
    def setUp(self):
        # Setup mocks for BaseRotatingClient
        self.mock_manager = MagicMock()
        self.mock_limit_resolver = MagicMock()
        
        # Instantiate BaseRotatingClient
        # We can instantiate it directly as we are only testing _is_rate_limit_error
        # which doesn't rely on abstract methods in __init__ for this test purpose,
        # but BaseRotatingClient is not abstract in python unless abc is used.
        # It does check HAS_OPENAI, so we assume openai is installed or mocked.
        self.client = BaseRotatingClient(
            manager=self.mock_manager,
            limit_resolver=self.mock_limit_resolver,
            default_model="test-model",
            provider="openai"
        )

        # Setup mocks for RotatingCredentialsMixin
        self.mock_wrapper = MagicMock()
        # RotatingCredentialsMixin expects keyword args including wrapper
        self.mixin = RotatingCredentialsMixin(
            model_id="test-model",
            wrapper=self.mock_wrapper
        )

    def test_openai_adapter_rate_limit_detection_error_1(self):
        """Test detection of the first specific 429 error format (OpenAI/Standard)."""
        # ERROR Rate limit error from OpenAI API: Error code: 429 - {'error': {'message': 'Rate limitexceeded: free-models-per-day...
        error_message = "Error code: 429 - {'error': {'message': 'Rate limitexceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests perday', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '50','X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1768348800000'}, 'provider_name':None}}, 'user_id': 'user_35Uqp8KORjNGZ6rcimjhpReCR1K'}"
        
        e = Exception(error_message)
        
        self.assertTrue(
            self.client._is_rate_limit_error(e), 
            f"Failed to detect Error 1 in OpenAI adapter. Error str: {str(e)}"
        )

    def test_openai_adapter_rate_limit_detection_error_2(self):
        """Test detection of the second specific 429 error format (Provider returned error / OpenRouter)."""
        # ERROR Rate limit error from OpenAI API: Error code: 429 - {'error': {'message': 'Providerreturned error', ...
        error_message = "Error code: 429 - {'error': {'message': 'Providerreturned error', 'code': 429, 'metadata': {'raw': 'qwen/qwen3-coder:free istemporarily rate-limited upstream. Please retry shortly, or add your own key toaccumulate your rate limits: https://openrouter.ai/settings/integrations','provider_name': 'Venice'}}, 'user_id': 'user_35WM9XjOiswqhxGPhAcjh7dCmq8'}"
        
        e = Exception(error_message)
        
        self.assertTrue(
            self.client._is_rate_limit_error(e), 
            f"Failed to detect Error 2 in OpenAI adapter. Error str: {str(e)}"
        )

    def test_mixin_rate_limit_detection_error_1(self):
        """Test detection of the first specific 429 error format in Mixin."""
        error_message = "Error code: 429 - {'error': {'message': 'Rate limitexceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests perday', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '50','X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1768348800000'}, 'provider_name':None}}, 'user_id': 'user_35Uqp8KORjNGZ6rcimjhpReCR1K'}"
        e = Exception(error_message)
        self.assertTrue(
            self.mixin._is_rate_limit_error(e), 
            "Failed to detect Error 1 in Mixin"
        )

    def test_mixin_rate_limit_detection_error_2(self):
        """
        Test the TRICKY case where str(e) does NOT contain '429', 
        but the body/metadata does.
        """
        # 1. The message is generic (this caused the original bug)
        generic_message = "Provider returned error"
        
        # 2. The details are in the body (reconstructed from your log)
        error_body = {
            'message': 'Provider returned error', 
            'code': 429, 
            'metadata': {
                'raw': 'qwen/qwen3-coder:free is temporarily rate-limited upstream...',
                'provider_name': 'Venice'
            }
        }
        
        # 3. Use the Mock class, NOT the base Exception class
        e = MockAPIError(generic_message, body=error_body)
        
        # Verify assumptions about the test setup itself
        print(f"\nDebug: str(e) is: '{str(e)}'") # Should print 'Provider returned error'
        assert "429" not in str(e), "Test setup flaw: 429 shouldn't be in the string representation"
        
        # 4. Run the check
        self.assertTrue(
            self.mixin._is_rate_limit_error(e), 
            "Failed to detect Error 2 via body inspection"
        )

    def test_status_code_detection(self):
        """Test detection via status_code attribute."""
        e = MockAPIError("Some error", status_code=429)
        self.assertTrue(self.client._is_rate_limit_error(e), "Client failed status_code check")
        self.assertTrue(self.mixin._is_rate_limit_error(e), "Mixin failed status_code check")

    def test_body_detection(self):
        """Test detection via body content."""
        e = MockAPIError("Generic error", body={"message": "You are rate limited"})
        self.assertTrue(self.client._is_rate_limit_error(e), "Client failed body check")
        self.assertTrue(self.mixin._is_rate_limit_error(e), "Mixin failed body check")

    def test_keyword_variations(self):
        """Test various keywords that should trigger rate limit detection."""
        keywords = [
            "too many requests", 
            "resource exhausted", 
            "rate limit", 
            "rate-limited"
        ]
        
        for kw in keywords:
            e = Exception(f"Some prefix {kw} some suffix")
            self.assertTrue(
                self.client._is_rate_limit_error(e), 
                f"Client failed to detect keyword: {kw}"
            )
            self.assertTrue(
                self.mixin._is_rate_limit_error(e), 
                f"Mixin failed to detect keyword: {kw}"
            )

if __name__ == '__main__':
    unittest.main()
