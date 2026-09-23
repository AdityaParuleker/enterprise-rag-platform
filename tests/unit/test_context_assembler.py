"""
Unit tests for ContextAssembler (Checkpoint 4.6 — Basic RAG Context Assembly).
"""

import uuid
import pytest
from backend.app.generation.context_assembler import ContextAssembler, DEFAULT_MAX_CONTEXT_CHARS


def create_sample_chunk(
    index: int = 1,
    text: str = "Sample chunk text for testing context assembly.",
    doc_id: str = None,
    doc_version: int = 1,
    page: int = 1,
    section: str = "Intro",
    score: float = 0.85
) -> dict:
    return {
        "chunk_id": f"00000000-0000-0000-0000-{index:012d}",
        "document_id": doc_id or str(uuid.uuid4()),
        "document_version": doc_version,
        "text": text,
        "page_number": page,
        "section_path": section,
        "metadata": {"source": "unit_test"},
        "distance": 1.0 - score,
        "similarity_score": score,
    }


def test_empty_retrieval_results():
    assembler = ContextAssembler()
    result = assembler.assemble([])

    assert result["context_text"] == ""
    assert result["included_chunks"] == []
    assert result["excluded_chunks"] == []
    assert result["total_chars"] == 0
    assert result["truncated_by_budget"] is False


def test_single_chunk_context():
    assembler = ContextAssembler()
    chunk = create_sample_chunk(1, text="Single chunk content.")
    result = assembler.assemble([chunk])

    assert result["total_chars"] <= DEFAULT_MAX_CONTEXT_CHARS
    assert len(result["included_chunks"]) == 1
    assert result["included_chunks"][0] == chunk
    assert result["excluded_chunks"] == []
    assert result["truncated_by_budget"] is False
    assert "Single chunk content." in result["context_text"]
    assert f"Chunk ID: {chunk['chunk_id']}" in result["context_text"]


def test_multiple_chunks_within_budget():
    assembler = ContextAssembler()
    chunk1 = create_sample_chunk(1, text="First chunk content.")
    chunk2 = create_sample_chunk(2, text="Second chunk content.")
    result = assembler.assemble([chunk1, chunk2])

    assert len(result["included_chunks"]) == 2
    assert result["included_chunks"] == [chunk1, chunk2]
    assert result["excluded_chunks"] == []
    assert result["truncated_by_budget"] is False
    assert "First chunk content." in result["context_text"]
    assert "Second chunk content." in result["context_text"]


def test_deterministic_ranked_ordering():
    assembler = ContextAssembler()
    chunk1 = create_sample_chunk(1, text="Rank 1 text", score=0.95)
    chunk2 = create_sample_chunk(2, text="Rank 2 text", score=0.85)
    chunk3 = create_sample_chunk(3, text="Rank 3 text", score=0.75)

    result = assembler.assemble([chunk1, chunk2, chunk3])
    pos1 = result["context_text"].find("Rank 1 text")
    pos2 = result["context_text"].find("Rank 2 text")
    pos3 = result["context_text"].find("Rank 3 text")

    assert pos1 < pos2 < pos3


def test_budget_boundary_stops_before_exceeding():
    max_chars = 600
    assembler = ContextAssembler(max_chars=max_chars)

    chunk1 = create_sample_chunk(1, text="Short chunk 1 text.")
    chunk2 = create_sample_chunk(2, text="Short chunk 2 text.")
    chunk3 = create_sample_chunk(3, text="Chunk 3 text that would push total length over 600 characters." * 10)

    result = assembler.assemble([chunk1, chunk2, chunk3])

    assert result["total_chars"] <= max_chars
    assert len(result["included_chunks"]) == 2
    assert result["included_chunks"] == [chunk1, chunk2]
    assert len(result["excluded_chunks"]) == 1
    assert result["excluded_chunks"] == [chunk3]
    assert result["truncated_by_budget"] is True


def test_no_partial_chunk_slicing():
    max_chars = 300
    assembler = ContextAssembler(max_chars=max_chars)

    chunk1 = create_sample_chunk(1, text="Chunk 1 full content.")
    chunk2 = create_sample_chunk(2, text="A very long chunk 2 text " * 10)

    result = assembler.assemble([chunk1, chunk2])

    assert result["total_chars"] <= max_chars
    assert "A very long chunk 2 text" not in result["context_text"]
    assert len(result["included_chunks"]) == 1
    assert result["included_chunks"][0] == chunk1
    assert len(result["excluded_chunks"]) == 1
    assert result["excluded_chunks"][0] == chunk2


def test_metadata_preservation():
    assembler = ContextAssembler()
    doc_id = str(uuid.uuid4())
    chunk = create_sample_chunk(
        index=10,
        text="Metadata verification text.",
        doc_id=doc_id,
        doc_version=3,
        page=42,
        section="Chapter 3 > Section 2",
        score=0.91,
    )

    result = assembler.assemble([chunk])
    inc_chunk = result["included_chunks"][0]

    assert inc_chunk["chunk_id"] == chunk["chunk_id"]
    assert inc_chunk["document_id"] == doc_id
    assert inc_chunk["document_version"] == 3
    assert inc_chunk["page_number"] == 42
    assert inc_chunk["section_path"] == "Chapter 3 > Section 2"
    assert inc_chunk["similarity_score"] == 0.91
    assert inc_chunk["metadata"] == {"source": "unit_test"}


def test_single_oversized_chunk_behavior():
    max_chars = 300
    assembler = ContextAssembler(max_chars=max_chars)

    oversized_text = "This single chunk is exceptionally long and will exceed the small context budget. " * 10
    oversized_chunk = create_sample_chunk(1, text=oversized_text)

    result = assembler.assemble([oversized_chunk])

    assert result["context_text"] == ""
    assert result["included_chunks"] == []
    assert len(result["excluded_chunks"]) == 1
    assert result["excluded_chunks"][0] == oversized_chunk
    assert result["total_chars"] == 0
    assert result["truncated_by_budget"] is True


def test_deterministic_repeatability():
    assembler = ContextAssembler()
    chunks = [create_sample_chunk(i, text=f"Chunk text {i}") for i in range(1, 5)]

    res1 = assembler.assemble(chunks)
    res2 = assembler.assemble(chunks)

    assert res1["context_text"] == res2["context_text"]
    assert res1["included_chunks"] == res2["included_chunks"]
    assert res1["excluded_chunks"] == res2["excluded_chunks"]
    assert res1["total_chars"] == res2["total_chars"]
    assert res1["truncated_by_budget"] == res2["truncated_by_budget"]


def test_configuration_driven_context_limit(monkeypatch):
    monkeypatch.setenv("RAG_MAX_CONTEXT_CHARS", "600")
    assembler = ContextAssembler()
    assert assembler.max_chars == 600

    custom_assembler = ContextAssembler(max_chars=1500)
    assert custom_assembler.max_chars == 1500
