"""
Phase 1 Unit Test: Provider Factory Verification
Verifies LLM_PROVIDER env variable resolution and unsupported provider exception behavior.
"""

import os
import pytest
from unittest.mock import patch
from backend.app.generation.providers.factory import get_llm_provider
from backend.app.generation.providers.ollama import OllamaProvider


def test_factory_returns_ollama_provider_for_ollama():
    """Verify factory returns OllamaProvider instance when LLM_PROVIDER is 'ollama'."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
        provider = get_llm_provider()
        assert isinstance(provider, OllamaProvider)


def test_factory_raises_value_error_for_unsupported_providers():
    """Verify factory raises ValueError for unsupported/unimplemented LLM_PROVIDER values (no silent fallback)."""
    unsupported_types = ["anthropic", "openai", "invalid_provider", "unknown"]
    for provider_name in unsupported_types:
        with patch.dict(os.environ, {"LLM_PROVIDER": provider_name}):
            with pytest.raises(ValueError) as exc_info:
                get_llm_provider()
            assert f"Unsupported LLM_PROVIDER: '{provider_name}'" in str(exc_info.value)
