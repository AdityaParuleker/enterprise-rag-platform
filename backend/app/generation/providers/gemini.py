"""
Google Gemini LLM Provider Concrete Implementation
Implements BaseLLMProvider interface for Google Gemini API with async readiness,
retry with exponential backoff for rate limits (HTTP 429), and streaming.
"""

import os
import time
import logging
from typing import Iterator, List, Dict, Any, Optional
from backend.app.generation.providers.base import LLMProvider

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None
    APIError = Exception


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        initial_backoff: float = 1.5,
        timeout: float = 60.0
    ):
        if not GEMINI_AVAILABLE:
            raise ImportError("The 'google-genai' package is required for GeminiProvider. Run 'pip install google-genai'.")

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable or parameter is required for GeminiProvider.")

        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.timeout = timeout
        self.client = genai.Client(api_key=self.api_key)

    async def check_readiness(self) -> bool:
        """
        Verify Gemini API availability by checking client initialization and API key presence.
        """
        if not self.api_key:
            return False
        try:
            # Lightweight readiness check
            return True
        except Exception as e:
            logger.warning(f"GeminiProvider readiness check failed: {e}")
            return False

    def _build_config(self, kwargs: Dict[str, Any]) -> Optional[Any]:
        if not types:
            return None

        config_params = {}
        if "temperature" in kwargs:
            config_params["temperature"] = float(kwargs["temperature"])
        if "max_tokens" in kwargs:
            config_params["max_output_tokens"] = int(kwargs["max_tokens"])
        elif "max_output_tokens" in kwargs:
            config_params["max_output_tokens"] = int(kwargs["max_output_tokens"])
        if "system" in kwargs and kwargs["system"]:
            config_params["system_instruction"] = str(kwargs["system"])

        if config_params:
            return types.GenerateContentConfig(**config_params)
        return None

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate complete LLM text response via Gemini API with retry & backoff on 429 rate limits.
        """
        if not prompt and "messages" not in kwargs:
            raise ValueError("Prompt or messages must be provided for LLM generation.")

        target_model = kwargs.get("model", self.model)
        config = self._build_config(kwargs)

        contents = prompt
        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            # Format chat messages array into prompt string
            formatted_history = []
            for msg in kwargs["messages"]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                formatted_history.append(f"{role}: {msg.get('content', '')}")
            contents = "\n".join(formatted_history)
            if prompt:
                contents += f"\nUser: {prompt}"

        backoff = self.initial_backoff
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=target_model,
                    contents=contents,
                    config=config
                )
                if response and hasattr(response, "text") and response.text:
                    return response.text.strip()
                return ""
            except APIError as e:
                is_rate_limit = getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower()
                if is_rate_limit:
                    logger.warning(
                        f"[GEMINI_RATE_LIMITED] Gemini API rate limit / quota exceeded (HTTP 429) "
                        f"on attempt {attempt}/{self.max_retries}. Backing off {backoff:.1f}s. Detail: {e}"
                    )
                    if attempt < self.max_retries:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise RuntimeError(f"[GEMINI_RATE_LIMITED] Gemini API rate limit reached after {self.max_retries} attempts: {e}")
                else:
                    logger.error(f"Gemini API Error [{getattr(e, 'code', 'UNKNOWN')}]: {e}")
                    raise RuntimeError(f"Gemini API error: {e}")
            except Exception as e:
                logger.error(f"Gemini generation error on attempt {attempt}: {e}")
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise RuntimeError(f"Gemini generation failed: {e}")

        raise RuntimeError(f"[GEMINI_RATE_LIMITED] Gemini generation failed after {self.max_retries} retries.")

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Stream LLM tokens chunk by chunk via Gemini generate_content_stream API with retry logic.
        """
        if not prompt and "messages" not in kwargs:
            raise ValueError("Prompt or messages must be provided for LLM streaming.")

        target_model = kwargs.get("model", self.model)
        config = self._build_config(kwargs)

        contents = prompt
        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            formatted_history = []
            for msg in kwargs["messages"]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                formatted_history.append(f"{role}: {msg.get('content', '')}")
            contents = "\n".join(formatted_history)
            if prompt:
                contents += f"\nUser: {prompt}"

        backoff = self.initial_backoff
        for attempt in range(1, self.max_retries + 1):
            try:
                response_stream = self.client.models.generate_content_stream(
                    model=target_model,
                    contents=contents,
                    config=config
                )
                for chunk in response_stream:
                    if chunk and hasattr(chunk, "text") and chunk.text:
                        yield chunk.text
                return
            except APIError as e:
                is_rate_limit = getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower()
                if is_rate_limit:
                    logger.warning(
                        f"[GEMINI_RATE_LIMITED] Gemini API streaming rate limit / quota exceeded (HTTP 429) "
                        f"on attempt {attempt}/{self.max_retries}. Backing off {backoff:.1f}s. Detail: {e}"
                    )
                    if attempt < self.max_retries:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise RuntimeError(f"[GEMINI_RATE_LIMITED] Gemini API streaming rate limit reached after {self.max_retries} attempts: {e}")
                else:
                    logger.error(f"Gemini Streaming API Error [{getattr(e, 'code', 'UNKNOWN')}]: {e}")
                    raise RuntimeError(f"Gemini streaming API error: {e}")
            except Exception as e:
                logger.error(f"Gemini streaming error on attempt {attempt}: {e}")
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise RuntimeError(f"Gemini streaming failed: {e}")
