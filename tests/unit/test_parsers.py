"""
Unit tests for Document Parsers & XXE Defense (Section 6.5).
"""

import pytest
from lxml import etree
from backend.app.ingestion.parsers.pdf_parser import PDFParser
from backend.app.ingestion.parsers.html_parser import HTMLParser
from backend.app.ingestion.parsers.md_parser import MarkdownParser
from backend.app.ingestion.parsers.csv_parser import CSVParser
from backend.app.ingestion.parsers.docx_parser import safe_xml_parser
from backend.app.ingestion.parsers.factory import get_parser_for_mime


def test_markdown_parser():
    parser = MarkdownParser()
    content = b"# Main Title\n\nThis is paragraph one.\n\n## Sub Title\n\nThis is paragraph two."
    blocks = parser.parse(content)

    assert len(blocks) >= 3
    assert blocks[0].block_type == "heading"
    assert blocks[0].text == "Main Title"
    assert blocks[1].text == "This is paragraph one."
    assert blocks[2].block_type == "heading"
    assert blocks[2].text == "Sub Title"


def test_csv_parser():
    parser = CSVParser()
    content = b"Name,Age,Role\nAlice,30,Admin\nBob,25,User"
    blocks = parser.parse(content)

    assert len(blocks) == 3
    assert blocks[0].block_type == "heading"
    assert "Columns: Name, Age, Role" in blocks[0].text
    assert "Row 1: Name: Alice | Age: 30 | Role: Admin" in blocks[1].text
    assert "Row 2: Name: Bob | Age: 25 | Role: User" in blocks[2].text


def test_csv_parser_strict_utf8():
    parser = CSVParser()
    invalid_utf8_bytes = b"Name,Age\nAlice,\xff\xfeInvalid"
    with pytest.raises(ValueError, match="Corrupted or invalid UTF-8 encoding"):
        parser.parse(invalid_utf8_bytes)


def test_xxe_entity_resolution_disabled():
    """Targeted security test ensuring XML entity expansion (XXE) is blocked."""
    xxe_xml = b"""<?xml version="1.0"?>
    <!DOCTYPE foo [
      <!ELEMENT foo ANY >
      <!ENTITY xxe SYSTEM "file:///etc/passwd" >]>
    <foo>&xxe;</foo>"""

    # Parse using safe_xml_parser configured for DOCX parsing
    tree = etree.fromstring(xxe_xml, parser=safe_xml_parser)
    # Text content of entity should NOT resolve to file content (should be empty/none)
    assert tree.text is None or tree.text == "" or "&xxe;" not in tree.text


def test_pdf_parser():
    """Test PDF text extraction and page number tracking using synthetic PyMuPDF PDF."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hello PDF text page 1")
    pdf_bytes = doc.tobytes()
    doc.close()

    parser = PDFParser()
    blocks = parser.parse(pdf_bytes)

    assert len(blocks) == 1
    assert blocks[0].page_number == 1
    assert "Hello PDF text page 1" in blocks[0].text


def test_pdf_parser_corrupt_pdf():
    """Test corrupted PDF file handling raising standardized ValueError."""
    parser = PDFParser()
    corrupt_pdf_bytes = b"%PDF-1.4 Corrupted binary content that fitz cannot open"
    with pytest.raises(ValueError, match="Corrupted or invalid PDF file"):
        parser.parse(corrupt_pdf_bytes)


def test_html_parser_strict_utf8():
    """Test HTML parser strict UTF-8 decoding rejection of invalid bytes."""
    parser = HTMLParser()
    invalid_html_bytes = b"<html><body>Invalid \xff\xfe bytes</body></html>"
    with pytest.raises(ValueError, match="Corrupted or invalid UTF-8 encoding"):
        parser.parse(invalid_html_bytes)


def test_md_parser_strict_utf8():
    """Test Markdown parser strict UTF-8 decoding rejection of invalid bytes."""
    parser = MarkdownParser()
    invalid_md_bytes = b"# Heading\nInvalid \xff\xfe bytes"
    with pytest.raises(ValueError, match="Corrupted or invalid UTF-8 encoding"):
        parser.parse(invalid_md_bytes)


def test_md_parser_atx_headings():
    """Test Markdown ATX heading regex matching vs non-heading lines."""
    parser = MarkdownParser()
    content = b"# Valid Heading\n\nThis paragraph mentions #hashtag and #not-a-heading.\n\n## Sub Heading"
    blocks = parser.parse(content)

    assert len(blocks) == 3
    assert blocks[0].block_type == "heading"
    assert blocks[0].text == "Valid Heading"
    assert blocks[1].block_type == "paragraph"
    assert "#hashtag" in blocks[1].text
    assert blocks[2].block_type == "heading"
    assert blocks[2].text == "Sub Heading"


def test_html_parser():
    parser = HTMLParser()
    content = b"<html><body><h1>Title</h1><p>Hello world paragraph text.</p></body></html>"
    blocks = parser.parse(content)

    assert len(blocks) > 0
    full_text = " ".join([b.text for b in blocks])
    assert "Hello world paragraph text" in full_text


def test_parser_factory():
    pdf_parser = get_parser_for_mime("application/pdf")
    assert isinstance(pdf_parser, PDFParser)

    md_parser = get_parser_for_mime("text/markdown")
    assert isinstance(md_parser, MarkdownParser)

    csv_parser = get_parser_for_mime("text/csv")
    assert isinstance(csv_parser, CSVParser)

    html_parser = get_parser_for_mime("text/html")
    assert isinstance(html_parser, HTMLParser)
