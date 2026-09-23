"""
Unit tests for Text Chunker (Section 6.5).
"""

from backend.app.ingestion.parsers.base import ParsedBlock
from backend.app.ingestion.chunkers.text_chunker import TextChunker, find_boundary_split


def test_text_chunker_linkage():
    chunker = TextChunker(target_size=500, overlap=50)
    blocks = [
        ParsedBlock(text="Paragraph 1 text content " * 15, section_title="Section A"),
        ParsedBlock(text="Paragraph 2 text content " * 15, section_title="Section B"),
        ParsedBlock(text="Paragraph 3 text content " * 15, section_title="Section C")
    ]

    doc_id = "doc-123"
    tenant_id = "tenant-456"

    chunks = chunker.chunk_blocks(blocks, document_id=doc_id, tenant_id=tenant_id)

    assert len(chunks) >= 2
    for idx, chunk in enumerate(chunks):
        assert chunk["document_id"] == doc_id
        assert chunk["tenant_id"] == tenant_id
        assert chunk["chunk_index"] == idx
        assert "approx_token_count" in chunk["metadata"]
        assert chunk["metadata"]["approx_token_count"] == len(chunk["text"]) // 4

        if idx == 0:
            assert chunk["prev_chunk_id"] is None
            assert chunk["next_chunk_id"] == chunks[1]["id"]
        elif idx == len(chunks) - 1:
            assert chunk["prev_chunk_id"] == chunks[idx - 1]["id"]
            assert chunk["next_chunk_id"] is None
        else:
            assert chunk["prev_chunk_id"] == chunks[idx - 1]["id"]
            assert chunk["next_chunk_id"] == chunks[idx + 1]["id"]


def test_max_chunk_size_enforced_on_huge_block():
    chunker = TextChunker(target_size=1500)
    # 5,000 char block
    huge_text = "This is a sentence inside a very large parsed block. " * 100
    blocks = [ParsedBlock(text=huge_text, section_title="Huge Section")]

    chunks = chunker.chunk_blocks(blocks, document_id="d1", tenant_id="t1")

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"]) <= 1800  # Strict guarantee: <= 1800 chars
        assert "approx_token_count" in chunk["metadata"]


def test_boundary_split_helper():
    text = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph content."
    split_idx = find_boundary_split(text, target_idx=30, window=150)
    assert split_idx > 0
    assert text[:split_idx].endswith("\n\n")


def test_text_chunker_empty_blocks():
    chunker = TextChunker()
    chunks = chunker.chunk_blocks([], document_id="d1", tenant_id="t1")
    assert chunks == []
