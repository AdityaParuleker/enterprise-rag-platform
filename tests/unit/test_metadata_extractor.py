"""
Unit tests for MetadataExtractor (Section 6.5).
"""

from backend.app.ingestion.metadata_extractor import MetadataExtractor
from backend.app.ingestion.parsers.base import ParsedBlock


def test_metadata_extractor_metrics():
    extractor = MetadataExtractor()
    content = "# Architecture Overview\n\nThis document describes the software system components.\nIt contains multiple lines and words."
    blocks = [ParsedBlock(text="Architecture Overview", block_type="heading")]

    metadata = extractor.extract_metadata(content=content, source_type="upload", filename="doc.md", blocks=blocks)

    assert metadata["title"] == "Architecture Overview"
    assert metadata["filename"] == "doc.md"
    assert metadata["source_type"] == "upload"
    assert metadata["char_count"] == len(content)
    assert metadata["word_count"] > 10
    assert metadata["approx_token_count"] == len(content) // 4
    assert "extracted_at" in metadata
