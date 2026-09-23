"""
Unit tests for OllamaEmbeddingProvider & Embedding Provider Factory (Checkpoint 4.2).
"""

import pytest
import httpx
from unittest.mock import patch, MagicMock
from backend.app.generation.providers.ollama import OllamaEmbeddingProvider
from backend.app.generation.providers.factory import get_embedding_provider


@pytest.mark.asyncio
async def test_check_readiness_success():
    provider = OllamaEmbeddingProvider(model="bge-large:latest")

    async def mock_get(url):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        if "/api/tags" in str(url):
            mock_resp.json.return_value = {
                "models": [{"name": "bge-large:latest"}]
            }
        return mock_resp

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        is_ready = await provider.check_readiness()
        assert is_ready is True


@pytest.mark.asyncio
async def test_check_readiness_ollama_unavailable():
    provider = OllamaEmbeddingProvider()

    with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):
        is_ready = await provider.check_readiness()
        assert is_ready is False


@pytest.mark.asyncio
async def test_check_readiness_model_not_installed():
    provider = OllamaEmbeddingProvider(model="bge-large:latest")

    async def mock_get(url):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        if "/api/tags" in str(url):
            mock_resp.json.return_value = {
                "models": [{"name": "llama2:latest"}]  # Missing bge-large
            }
        return mock_resp

    with patch("httpx.AsyncClient.get", side_effect=mock_get):
        is_ready = await provider.check_readiness()
        assert is_ready is False


@pytest.mark.asyncio
async def test_check_readiness_does_not_perform_inference():
    provider = OllamaEmbeddingProvider(model="bge-large:latest")

    async def mock_get(url):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "bge-large:latest"}]}
        return mock_resp

    with patch("httpx.AsyncClient.get", side_effect=mock_get), \
         patch("httpx.Client.post") as mock_post:
        
        is_ready = await provider.check_readiness()
        assert is_ready is True
        # Verify no POST embedding/inference request was made
        mock_post.assert_not_called()


def test_embed_single_success_1024_dims():
    provider = OllamaEmbeddingProvider(expected_dim=1024)
    fake_vector = [0.1] * 1024

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"embedding": fake_vector}

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        vector = provider.embed("sample text")
        assert len(vector) == 1024
        assert vector == fake_vector
        mock_post.assert_called_once()
        assert "api/embeddings" in mock_post.call_args[0][0]


def test_embed_single_model_not_found_404():
    provider = OllamaEmbeddingProvider(model="missing-model")

    mock_resp = MagicMock()
    mock_resp.status_code = 404

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(ValueError, match="is not installed in Ollama"):
            provider.embed("sample text")


def test_embed_single_dimension_mismatch():
    provider = OllamaEmbeddingProvider(expected_dim=1024)
    invalid_vector = [0.1] * 512  # Mismatch: 512 instead of 1024

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"embedding": invalid_vector}

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(ValueError, match="EMBEDDING_DIMENSION_MISMATCH"):
            provider.embed("sample text")


def test_embed_single_http_error():
    provider = OllamaEmbeddingProvider()

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="Ollama embedding request failed"):
            provider.embed("sample text")


def test_embed_batch_api_embed_success():
    provider = OllamaEmbeddingProvider(expected_dim=1024)
    vec1 = [0.1] * 1024
    vec2 = [0.2] * 1024

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"embeddings": [vec1, vec2]}

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        results = provider.embed_batch(["text1", "text2"])
        assert len(results) == 2
        assert results[0] == vec1
        assert results[1] == vec2
        assert "api/embed" in mock_post.call_args[0][0]


def test_embed_batch_fallback_sequential():
    provider = OllamaEmbeddingProvider(expected_dim=1024)
    vec1 = [0.1] * 1024

    def mock_post(*args, **kwargs):
        mock_resp = MagicMock()
        url_arg = str(args[1] if len(args) > 1 else args[0])
        if url_arg.endswith("/api/embed"):
            mock_resp.status_code = 404  # Batch endpoint not supported
        else:
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"embedding": vec1}
        return mock_resp

    with patch("httpx.Client.post", side_effect=mock_post):
        results = provider.embed_batch(["text1", "text2"])
        assert len(results) == 2
        assert results[0] == vec1
        assert results[1] == vec1


def test_embed_batch_fallback_dimension_mismatch():
    provider = OllamaEmbeddingProvider(expected_dim=1024)
    invalid_vec = [0.1] * 256

    def mock_post(*args, **kwargs):
        mock_resp = MagicMock()
        url_arg = str(args[1] if len(args) > 1 else args[0])
        if url_arg.endswith("/api/embed"):
            mock_resp.status_code = 404  # Trigger fallback
        else:
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"embedding": invalid_vec}
        return mock_resp

    with patch("httpx.Client.post", side_effect=mock_post):
        with pytest.raises(ValueError, match="EMBEDDING_DIMENSION_MISMATCH"):
            provider.embed_batch(["text1"])


def test_embedding_provider_factory(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    provider = get_embedding_provider()
    assert isinstance(provider, OllamaEmbeddingProvider)

    monkeypatch.setenv("EMBEDDING_PROVIDER", "unsupported_provider")
    with pytest.raises(ValueError, match="Unsupported EMBEDDING_PROVIDER"):
        get_embedding_provider()

