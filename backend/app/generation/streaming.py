"""
Server-Sent Events (SSE) Streaming Infrastructure (Phase 4 — Basic RAG)
Implements SSE event formatting, token/citation/done/error stream generator wrappers,
and HTTP StreamingResponse creation.
"""

import json
from typing import Iterator, AsyncIterator, List, Dict, Any, Optional, Union
from fastapi.responses import StreamingResponse

DEFAULT_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no"
}


def format_sse_event(event_type: str, payload: Dict[str, Any]) -> str:
    """
    Format a data dictionary into a standard W3C SSE event string.

    Format:
        data: {"type": event_type, ...}\n\n
    """
    event_data = {"type": event_type}
    event_data.update(payload)
    json_str = json.dumps(event_data, ensure_ascii=False)
    return f"data: {json_str}\n\n"


def format_token_event(content: str) -> str:
    """Format a single token SSE event."""
    return format_sse_event("token", {"content": content})


def format_citation_event(citation: Dict[str, Any]) -> str:
    """Format a single citation SSE event."""
    return format_sse_event("citation", {"citation": citation})


def format_error_event(error_message: str, code: str = "STREAMING_ERROR") -> str:
    """Format an error SSE event."""
    return format_sse_event("error", {"error": error_message, "code": code})


def format_done_event(metadata: Optional[Dict[str, Any]] = None) -> str:
    """Format a stream completion 'done' SSE event."""
    return format_sse_event("done", {"metadata": metadata or {}})


class ValidatedStreamer:
    """
    Buffer-Validate-Stream Generator wrapping LLM/context token streams into
    safe, formatted SSE events.
    """

    def stream_tokens(
        self,
        token_stream: Union[Iterator[str], AsyncIterator[str]],
        citations: Optional[List[Dict[str, Any]]] = None,
        done_metadata: Optional[Dict[str, Any]] = None
    ) -> Iterator[str]:
        """
        Synchronous generator converting raw token iterator into formatted SSE data lines.
        Emits tokens -> optional citations -> done event.
        Catches runtime exceptions and emits error event before closing stream.
        """
        try:
            for token in token_stream:
                if token:
                    yield format_token_event(token)

            if citations:
                for cit in citations:
                    yield format_citation_event(cit)

            yield format_done_event(done_metadata)
        except Exception as e:
            yield format_error_event(str(e), code="GENERATION_FAILED")

    async def astream_tokens(
        self,
        token_stream: Union[Iterator[str], AsyncIterator[str]],
        citations: Optional[List[Dict[str, Any]]] = None,
        done_metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """
        Async generator for FastAPI StreamingResponse converting async token stream
        into formatted SSE data lines.
        """
        try:
            if hasattr(token_stream, "__aiter__"):
                async for token in token_stream:
                    if token:
                        yield format_token_event(token)
            else:
                for token in token_stream:
                    if token:
                        yield format_token_event(token)

            if citations:
                for cit in citations:
                    yield format_citation_event(cit)

            yield format_done_event(done_metadata)
        except Exception as e:
            yield format_error_event(str(e), code="GENERATION_FAILED")

    def create_sse_response(
        self,
        generator: Union[Iterator[str], AsyncIterator[str]],
        headers: Optional[Dict[str, str]] = None
    ) -> StreamingResponse:
        """
        Create a FastAPI StreamingResponse configured for text/event-stream with required headers.
        """
        resp_headers = dict(DEFAULT_SSE_HEADERS)
        if headers:
            resp_headers.update(headers)

        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers=resp_headers
        )
