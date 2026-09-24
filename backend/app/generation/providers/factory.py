"""
LLM Provider Factory (Section 2)
Resolves configured LLM provider based on LLM_PROVIDER env variable.
Raises ValueError for unsupported or un-implemented provider types.
"""

import os
from backend.app.generation.providers.base import LLMProvider, EmbeddingProvider
from backend.app.generation.providers.ollama import OllamaProvider, OllamaEmbeddingProvider


def get_llm_provider() -> LLMProvider:
    provider_type = os.getenv("LLM_PROVIDER", "").lower()
    gemini_key = os.getenv("GEMINI_API_KEY")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    is_cloud_env = bool(os.getenv("VERCEL") or os.getenv("RENDER") or os.getenv("ENVIRONMENT") == "production")

    # Priority 1: Explicit or implicit Gemini configuration
    if provider_type in ("gemini", "google") or gemini_key:
        from backend.app.generation.providers.gemini import GeminiProvider
        return GeminiProvider()

    # Priority 2: Explicit Ollama or default local development
    if provider_type == "ollama" or not provider_type:
        if is_cloud_env and ("localhost" in ollama_url or "127.0.0.1" in ollama_url):
            raise RuntimeError(
                "Cloud deployment environment detected (Vercel/Render/production), but LLM_PROVIDER resolved to 'ollama' "
                "with an unreachable localhost URL (http://localhost:11434) and missing GEMINI_API_KEY. "
                "Please configure GEMINI_API_KEY or set LLM_PROVIDER=gemini in environment variables."
            )
        return OllamaProvider()

    raise ValueError(f"Unsupported LLM_PROVIDER: '{provider_type}'")


def get_embedding_provider() -> EmbeddingProvider:
    provider_type = os.getenv("EMBEDDING_PROVIDER", "").lower()
    gemini_key = os.getenv("GEMINI_API_KEY")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    is_cloud_env = bool(os.getenv("VERCEL") or os.getenv("RENDER") or os.getenv("ENVIRONMENT") == "production")

    # Priority 1: Explicit or implicit Gemini embedding configuration
    if provider_type in ("gemini", "google") or gemini_key:
        from backend.app.generation.providers.gemini import GeminiEmbeddingProvider
        return GeminiEmbeddingProvider()

    # Priority 2: Explicit Ollama/bge-large or default local development
    if provider_type in ("ollama", "bge-large") or not provider_type:
        if is_cloud_env and ("localhost" in ollama_url or "127.0.0.1" in ollama_url):
            raise RuntimeError(
                "Cloud deployment environment detected (Vercel/Render/production), but EMBEDDING_PROVIDER resolved to 'ollama' "
                "with an unreachable localhost URL (http://localhost:11434) and missing GEMINI_API_KEY. "
                "Please configure GEMINI_API_KEY or set EMBEDDING_PROVIDER=gemini in environment variables."
            )
        return OllamaEmbeddingProvider()

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: '{provider_type}'")


def validate_provider_configs() -> None:
    """
    Fail-fast startup validation for LLM and Embedding providers.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    llm_provider = os.getenv("LLM_PROVIDER", "gemini" if gemini_key else "ollama").lower()
    allowed_llm = {"ollama", "gemini", "google"}
    if llm_provider not in allowed_llm:
        raise ValueError(
            f"Invalid LLM_PROVIDER '{llm_provider}'. Supported LLM providers: {sorted(list(allowed_llm))}"
        )

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "gemini" if gemini_key else "ollama").lower()
    allowed_embedding = {"ollama", "bge-large", "gemini", "google"}
    if embedding_provider not in allowed_embedding:
        raise ValueError(
            f"Invalid EMBEDDING_PROVIDER '{embedding_provider}'. Supported EMBEDDING providers: {sorted(list(allowed_embedding))}"
        )




