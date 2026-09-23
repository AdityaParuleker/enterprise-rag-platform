"""
LLM Provider Factory (Section 2)
Resolves configured LLM provider based on LLM_PROVIDER env variable.
Raises ValueError for unsupported or un-implemented provider types.
"""

import os
from backend.app.generation.providers.base import LLMProvider, EmbeddingProvider
from backend.app.generation.providers.ollama import OllamaProvider, OllamaEmbeddingProvider


def get_llm_provider() -> LLMProvider:
    provider_type = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider_type == "ollama":
        return OllamaProvider()
    elif provider_type in ("gemini", "google"):
        from backend.app.generation.providers.gemini import GeminiProvider
        return GeminiProvider()
    
    raise ValueError(f"Unsupported LLM_PROVIDER: '{provider_type}'")


def get_embedding_provider() -> EmbeddingProvider:
    provider_type = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
    if provider_type in ("ollama", "bge-large"):
        return OllamaEmbeddingProvider()
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: '{provider_type}'")


def validate_provider_configs() -> None:
    """
    Fail-fast startup validation for LLM and Embedding providers.
    Uses separate allow-lists to ensure unsupported providers (e.g. gemini embeddings) fail loudly at worker boot.
    """
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    allowed_llm = {"ollama", "gemini", "google"}
    if llm_provider not in allowed_llm:
        raise ValueError(
            f"Invalid LLM_PROVIDER '{llm_provider}'. Supported LLM providers: {sorted(list(allowed_llm))}"
        )

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
    allowed_embedding = {"ollama", "bge-large"}
    if embedding_provider not in allowed_embedding:
        raise ValueError(
            f"Invalid EMBEDDING_PROVIDER '{embedding_provider}'. Supported EMBEDDING providers: {sorted(list(allowed_embedding))}"
        )


