"""
Ollama Provider Concrete Implementation (Section 2 & Phase 1 Readiness)
Implements endpoint availability readiness check — generate() and stream() remain stubs.
"""

import os
import httpx
from typing import Iterator, List
from backend.app.generation.providers.base import LLMProvider, EmbeddingProvider


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = None, model: str = None, timeout: float = 180.0):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "llama3.2:1b")
        self.timeout = timeout

    async def check_readiness(self) -> bool:
        """Verify configured Ollama base URL availability without model inference."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(self.base_url)
                return response.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate full LLM response via Ollama /api/generate or /api/chat.
        """
        if not prompt and "messages" not in kwargs:
            raise ValueError("Prompt or messages must be provided for LLM generation.")

        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", 0.0)
        req_timeout = kwargs.get("timeout", self.timeout)

        options = kwargs.get("options", {}).copy() if isinstance(kwargs.get("options"), dict) else {}
        options.setdefault("temperature", temperature)
        if "max_tokens" in kwargs and "num_predict" not in options:
            options["num_predict"] = kwargs["max_tokens"]
        elif "num_predict" in kwargs and "num_predict" not in options:
            options["num_predict"] = kwargs["num_predict"]

        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": model,
                "messages": kwargs["messages"],
                "stream": False,
                "options": options
            }
        else:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": options
            }
            if "system" in kwargs:
                payload["system"] = kwargs["system"]

        try:
            with httpx.Client(timeout=req_timeout) as client:
                resp = client.post(url, json=payload)
        except Exception as e:
            raise RuntimeError(f"Ollama connection error: {str(e)}")

        if resp.status_code == 404:
            raise ValueError(f"Configured LLM model '{model}' is not installed in Ollama.")
        elif resp.status_code != 200:
            raise RuntimeError(f"Ollama generation request failed [{resp.status_code}]: {resp.text}")

        data = resp.json()
        if "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content", "")
        return data.get("response", "")

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Stream LLM tokens chunk by chunk via Ollama /api/generate or /api/chat ndjson streaming endpoint.
        """
        if not prompt and "messages" not in kwargs:
            raise ValueError("Prompt or messages must be provided for LLM streaming.")

        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", 0.0)

        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": model,
                "messages": kwargs["messages"],
                "stream": True,
                "options": {"temperature": temperature}
            }
            is_chat = True
        else:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": temperature}
            }
            if "system" in kwargs:
                payload["system"] = kwargs["system"]
            is_chat = False

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code == 404:
                        raise ValueError(f"Configured LLM model '{model}' is not installed in Ollama.")
                    elif response.status_code != 200:
                        raise RuntimeError(f"Ollama streaming request failed [{response.status_code}]")

                    import json
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if is_chat:
                            chunk = item.get("message", {}).get("content", "")
                        else:
                            chunk = item.get("response", "")

                        if chunk:
                            yield chunk

                        if item.get("done", False):
                            break
        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            raise RuntimeError(f"Ollama streaming HTTP error: {str(e)}")


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, base_url: str = None, model: str = None, expected_dim: int = None):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("EMBEDDING_MODEL", "bge-large:latest")
        self.expected_dim = expected_dim or int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

    async def check_readiness(self) -> bool:
        """
        Verify configured Ollama endpoint reachability AND configured embedding model availability
        via /api/tags WITHOUT performing embedding inference.
        """
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                base_resp = await client.get(self.base_url)
                if base_resp.status_code != 200:
                    return False

                tags_resp = await client.get(f"{self.base_url}/api/tags")
                if tags_resp.status_code != 200:
                    return False

                tags_data = tags_resp.json()
                models = [m.get("name", "") for m in tags_data.get("models", [])]
                
                target = self.model.lower()
                target_base = target.split(":")[0]
                model_found = any(
                    m.lower() == target or m.lower().split(":")[0] == target_base or target_base in m.lower()
                    for m in models
                )
                return model_found
        except Exception:
            return False

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single string using Ollama /api/embeddings.
        Validates output dimension == expected_dim (1024).
        """
        if not text:
            text = " "  # Fallback non-empty string

        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(url, json=payload)
        except Exception as e:
            raise RuntimeError(f"Ollama embedding HTTP connection error: {str(e)}")

        if response.status_code == 404:
            raise ValueError(f"Configured embedding model '{self.model}' is not installed in Ollama.")
        elif response.status_code != 200:
            raise RuntimeError(f"Ollama embedding request failed [{response.status_code}]: {response.text}")

        data = response.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("Ollama embedding response missing 'embedding' array")

        if len(embedding) != self.expected_dim:
            raise ValueError(f"EMBEDDING_DIMENSION_MISMATCH: expected {self.expected_dim}, got {len(embedding)}")

        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of strings.
        Validates all vector dimensions == expected_dim (1024).
        """
        if not texts:
            return []

        embed_url = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": texts}
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(embed_url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = data.get("embeddings")
                    if isinstance(embeddings, list) and len(embeddings) == len(texts):
                        for vec in embeddings:
                            if len(vec) != self.expected_dim:
                                raise ValueError(f"EMBEDDING_DIMENSION_MISMATCH: expected {self.expected_dim}, got {len(vec)}")
                        return embeddings
        except ValueError:
            raise
        except Exception:
            pass

        batch_results: List[List[float]] = []
        for text in texts:
            batch_results.append(self.embed(text))
        return batch_results

