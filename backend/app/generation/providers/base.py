"""
LLM and Embedding Provider Abstract Interfaces (Section 2 Verbatim)
Method signatures and return types only — provider contract with async check_readiness.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List


class LLMProvider(ABC):
    @abstractmethod
    async def check_readiness(self) -> bool:
        """Asynchronously check provider endpoint availability (availability-only, no inference)."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        raise NotImplementedError


class EmbeddingProvider(ABC):
    @abstractmethod
    async def check_readiness(self) -> bool:
        """Asynchronously check provider endpoint and model availability (no inference)."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError
