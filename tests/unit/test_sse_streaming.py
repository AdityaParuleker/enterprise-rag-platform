"""
Unit tests for SSE Streaming Infrastructure (Checkpoint 4.9 — Basic RAG).
"""

import json
import pytest
from backend.app.generation.streaming import (
    format_sse_event,
    format_token_event,
    format_citation_event,
    format_error_event,
    format_done_event,
    ValidatedStreamer,
    DEFAULT_SSE_HEADERS,
)


def test_format_sse_event_helpers():
    token_str = format_token_event("Hello world")
    assert token_str.startswith("data: ")
    assert token_str.endswith("\n\n")
    data_token = json.loads(token_str[6:-2])
    assert data_token["type"] == "token"
    assert data_token["content"] == "Hello world"

    cit = {"chunk_id": "c1", "document_id": "d1"}
    cit_str = format_citation_event(cit)
    data_cit = json.loads(cit_str[6:-2])
    assert data_cit["type"] == "citation"
    assert data_cit["citation"] == cit

    err_str = format_error_event("Generation failed", code="ERR_CODE")
    data_err = json.loads(err_str[6:-2])
    assert data_err["type"] == "error"
    assert data_err["error"] == "Generation failed"
    assert data_err["code"] == "ERR_CODE"

    done_str = format_done_event({"tokens": 10})
    data_done = json.loads(done_str[6:-2])
    assert data_done["type"] == "done"
    assert data_done["metadata"] == {"tokens": 10}


def test_stream_tokens_sync():
    streamer = ValidatedStreamer()
    tokens = ["The ", "quick ", "brown ", "fox"]
    citations = [{"chunk_id": "c1"}]
    metadata = {"finish_reason": "stop"}

    events = list(streamer.stream_tokens(tokens, citations=citations, done_metadata=metadata))

    assert len(events) == 6  # 4 tokens + 1 citation + 1 done
    parsed = [json.loads(e[6:-2]) for e in events]

    assert parsed[0] == {"type": "token", "content": "The "}
    assert parsed[1] == {"type": "token", "content": "quick "}
    assert parsed[2] == {"type": "token", "content": "brown "}
    assert parsed[3] == {"type": "token", "content": "fox"}
    assert parsed[4] == {"type": "citation", "citation": {"chunk_id": "c1"}}
    assert parsed[5] == {"type": "done", "metadata": {"finish_reason": "stop"}}


@pytest.mark.asyncio
async def test_stream_tokens_async():
    streamer = ValidatedStreamer()

    async def token_gen():
        yield "Async "
        yield "token"

    citations = [{"chunk_id": "c1"}]
    events = []
    async for event in streamer.astream_tokens(token_gen(), citations=citations):
        events.append(event)

    assert len(events) == 4  # 2 tokens + 1 citation + 1 done
    parsed = [json.loads(e[6:-2]) for e in events]

    assert parsed[0] == {"type": "token", "content": "Async "}
    assert parsed[1] == {"type": "token", "content": "token"}
    assert parsed[2] == {"type": "citation", "citation": {"chunk_id": "c1"}}
    assert parsed[3] == {"type": "done", "metadata": {}}


def test_stream_tokens_handles_exception():
    streamer = ValidatedStreamer()

    def faulty_gen():
        yield "Token 1"
        raise RuntimeError("LLM network disconnected")

    events = list(streamer.stream_tokens(faulty_gen()))

    assert len(events) == 2  # 1 token + 1 error
    parsed1 = json.loads(events[0][6:-2])
    parsed2 = json.loads(events[1][6:-2])

    assert parsed1 == {"type": "token", "content": "Token 1"}
    assert parsed2["type"] == "error"
    assert parsed2["error"] == "LLM network disconnected"
    assert parsed2["code"] == "GENERATION_FAILED"


def test_create_sse_response():
    streamer = ValidatedStreamer()

    def dummy_gen():
        yield format_token_event("Test")

    response = streamer.create_sse_response(dummy_gen())

    assert response.media_type == "text/event-stream"
    assert response.headers["Cache-Control"] == "no-cache"
    assert response.headers["Connection"] == "keep-alive"
    assert response.headers["X-Accel-Buffering"] == "no"
