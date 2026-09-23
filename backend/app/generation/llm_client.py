"""
LLM Client Adapter Module (Phase 4 — Basic RAG)
Provides high-level LLM adapter delegating to underlying LLMProvider instance.
"""

from typing import Iterator, Optional
from backend.app.generation.providers.base import LLMProvider
from backend.app.generation.providers.factory import get_llm_provider


class LLMClient:
    """
    Adapter client wrapping configured LLMProvider for generation and streaming.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        provider_name: Optional[str] = None
    ):
        if provider is not None:
            self.provider = provider
        elif provider_name is not None and isinstance(provider_name, str):
            # If provider_name is passed as string, resolve via factory
            self.provider = get_llm_provider()
        else:
            self.provider = get_llm_provider()

        self.provider_name = provider_name or self.provider.__class__.__name__

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate complete LLM text response.
        """
        return self.provider.generate(prompt, **kwargs)

    def stream_response(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Stream LLM tokens chunk by chunk.
        """
        return self.provider.stream(prompt, **kwargs)

