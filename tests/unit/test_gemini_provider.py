"""
Unit tests for GeminiProvider & factory resolution.
Tests readiness, generation, streaming, 429 rate limit backoff logging, and factory integration.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from backend.app.generation.providers.factory import get_llm_provider
from backend.app.generation.providers.gemini import GeminiProvider, GEMINI_AVAILABLE


@pytest.mark.skipif(not GEMINI_AVAILABLE, reason="google-genai package not installed")
def test_factory_gemini_resolution(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key_for_testing")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    provider = get_llm_provider()
    assert isinstance(provider, GeminiProvider)
    assert provider.model == "gemini-3.6-flash"



@pytest.mark.skipif(not GEMINI_AVAILABLE, reason="google-genai package not installed")
def test_gemini_generate_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key_for_testing")
    provider = GeminiProvider(api_key="dummy_key_for_testing", model="gemini-3.6-flash")
    
    mock_response = MagicMock()
    mock_response.text = " Hello from Gemini "
    
    with patch.object(provider.client.models, "generate_content", return_value=mock_response) as mock_gen:
        res = provider.generate("Say hello", system="Be concise")
        assert res == "Hello from Gemini"
        mock_gen.assert_called_once()


@pytest.mark.skipif(not GEMINI_AVAILABLE, reason="google-genai package not installed")
def test_gemini_stream_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key_for_testing")
    provider = GeminiProvider(api_key="dummy_key_for_testing", model="gemini-3.6-flash")

    chunk1 = MagicMock()
    chunk1.text = "Hello "
    chunk2 = MagicMock()
    chunk2.text = "world!"

    with patch.object(provider.client.models, "generate_content_stream", return_value=[chunk1, chunk2]):
        tokens = list(provider.stream("Stream this"))
        assert tokens == ["Hello ", "world!"]


@pytest.mark.skipif(not GEMINI_AVAILABLE, reason="google-genai package not installed")
def test_gemini_rate_limit_backoff(caplog, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key_for_testing")
    provider = GeminiProvider(api_key="dummy_key_for_testing", max_retries=2, initial_backoff=0.01)

    from google.genai.errors import APIError
    error_429 = APIError(429, {"error": {"message": "RESOURCE_EXHAUSTED: Quota exceeded"}})

    with patch.object(provider.client.models, "generate_content", side_effect=error_429):
        with pytest.raises(RuntimeError) as exc_info:
            provider.generate("Test prompt")

        assert "[GEMINI_RATE_LIMITED]" in str(exc_info.value)
        assert "[GEMINI_RATE_LIMITED]" in caplog.text
