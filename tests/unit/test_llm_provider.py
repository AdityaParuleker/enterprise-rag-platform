"""
Unit tests for OllamaProvider, LLMClient, and LLMProvider Factory (Checkpoint 4.8 — Basic RAG).
"""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from backend.app.generation.providers.ollama import OllamaProvider
from backend.app.generation.llm_client import LLMClient
from backend.app.generation.providers.factory import get_llm_provider


@pytest.mark.asyncio
async def test_ollama_provider_check_readiness_success():
    provider = OllamaProvider(base_url="http://localhost:11434")

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        result = await provider.check_readiness()
        assert result is True
        mock_get.assert_called_once_with("http://localhost:11434")


@pytest.mark.asyncio
async def test_ollama_provider_check_readiness_failure():
    provider = OllamaProvider(base_url="http://localhost:11434")

    with patch("httpx.AsyncClient.get", side_effect=Exception("Connection refused")):
        result = await provider.check_readiness()
        assert result is False


def test_ollama_provider_generate_prompt_success():
    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3.2:1b")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": "This is the generated answer.", "done": True}

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        ans = provider.generate("Summarize the policy document.")
        assert ans == "This is the generated answer."
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "llama3.2:1b"
        assert call_kwargs["json"]["prompt"] == "Summarize the policy document."


def test_ollama_provider_generate_messages_chat_success():
    provider = OllamaProvider(base_url="http://localhost:11434")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {"role": "assistant", "content": "Chat assistant response."},
        "done": True
    }

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ]

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        ans = provider.generate("", messages=messages)
        assert ans == "Chat assistant response."
        call_url = mock_post.call_args[0][0]
        assert "/api/chat" in call_url


def test_ollama_provider_generate_model_not_installed_404():
    provider = OllamaProvider()

    mock_resp = MagicMock()
    mock_resp.status_code = 404

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(ValueError, match="is not installed in Ollama"):
            provider.generate("Test prompt")


def test_ollama_provider_generate_http_error_500():
    provider = OllamaProvider()

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    with patch("httpx.Client.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="Ollama generation request failed"):
            provider.generate("Test prompt")


def test_ollama_provider_generate_empty_input_raises_value_error():
    provider = OllamaProvider()
    with pytest.raises(ValueError, match="Prompt or messages must be provided"):
        provider.generate("")


def test_ollama_provider_stream_success():
    provider = OllamaProvider()

    lines = [
        b'{"response": "Hello", "done": false}',
        b'{"response": " world", "done": false}',
        b'{"response": "!", "done": true}',
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = lines

    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_resp
    mock_context.__exit__.return_value = None

    with patch("httpx.Client.stream", return_value=mock_context):
        chunks = list(provider.stream("Stream prompt"))
        assert chunks == ["Hello", " world", "!"]


def test_ollama_provider_stream_chat_messages_success():
    provider = OllamaProvider()

    lines = [
        b'{"message": {"role": "assistant", "content": "Token1"}, "done": false}',
        b'{"message": {"role": "assistant", "content": "Token2"}, "done": true}',
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = lines

    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_resp
    mock_context.__exit__.return_value = None

    messages = [{"role": "user", "content": "Stream chat"}]

    with patch("httpx.Client.stream", return_value=mock_context):
        chunks = list(provider.stream("", messages=messages))
        assert chunks == ["Token1", "Token2"]


def test_ollama_provider_stream_http_404():
    provider = OllamaProvider()

    mock_resp = MagicMock()
    mock_resp.status_code = 404

    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_resp
    mock_context.__exit__.return_value = None

    with patch("httpx.Client.stream", return_value=mock_context):
        with pytest.raises(ValueError, match="is not installed in Ollama"):
            list(provider.stream("Stream prompt"))


def test_llm_client_adapter_delegation():
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "Adapter generated response"
    mock_provider.stream.return_value = iter(["TokenA", "TokenB"])

    client = LLMClient(provider=mock_provider)

    assert client.generate_response("Test prompt") == "Adapter generated response"
    mock_provider.generate.assert_called_once_with("Test prompt")

    tokens = list(client.stream_response("Test prompt"))
    assert tokens == ["TokenA", "TokenB"]
    mock_provider.stream.assert_called_once_with("Test prompt")


def test_llm_provider_factory(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    provider = get_llm_provider()
    assert isinstance(provider, OllamaProvider)

    monkeypatch.setenv("LLM_PROVIDER", "unsupported_provider")
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        get_llm_provider()
