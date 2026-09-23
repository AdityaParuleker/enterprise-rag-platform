"""
Unit tests for Ingestion Security & Magic-Byte Validation (Section 6.5).
"""

import pytest
from backend.app.ingestion.security import (
    IngestionSecurityValidator,
    SecurityValidationError,
    sanitize_filename,
    _is_ip_allowed
)


def test_sanitize_filename():
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("test\x00file.pdf") == "testfile.pdf"
    assert sanitize_filename("C:\\Windows\\System32\\cmd.exe") == "cmd.exe"
    assert sanitize_filename("") == "unnamed_document"


def test_ip_allowed_checks():
    assert _is_ip_allowed("127.0.0.1") is False
    assert _is_ip_allowed("10.0.0.5") is False
    assert _is_ip_allowed("172.16.0.1") is False
    assert _is_ip_allowed("192.168.1.1") is False
    assert _is_ip_allowed("169.254.169.254") is False
    assert _is_ip_allowed("0.0.0.0") is False
    assert _is_ip_allowed("8.8.8.8") is True
    assert _is_ip_allowed("1.1.1.1") is True


def test_validate_file_bytes_pdf():
    validator = IngestionSecurityValidator()
    pdf_bytes = b"%PDF-1.5\n% \n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    mime, name = validator.validate_file_bytes(pdf_bytes, filename="test.pdf")
    assert mime == "application/pdf"
    assert name == "test.pdf"


def test_validate_file_bytes_html():
    validator = IngestionSecurityValidator()
    html_bytes = b"<!DOCTYPE html><html><head><title>Test</title></head><body><h1>Hello</h1></body></html>"
    mime, name = validator.validate_file_bytes(html_bytes, filename="page.html")
    assert mime == "text/html"
    assert name == "page.html"


def test_validate_file_bytes_markdown():
    validator = IngestionSecurityValidator()
    md_bytes = b"# Header 1\n\nThis is markdown text content."
    mime, name = validator.validate_file_bytes(md_bytes, filename="notes.md")
    assert mime == "text/markdown"
    assert name == "notes.md"


def test_validate_file_bytes_invalid_binary():
    validator = IngestionSecurityValidator()
    invalid_bytes = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"
    with pytest.raises(SecurityValidationError, match="binary null bytes|invalid file byte signature"):
        validator.validate_file_bytes(invalid_bytes, filename="bad.bin")


def test_validate_file_bytes_empty():
    validator = IngestionSecurityValidator()
    with pytest.raises(SecurityValidationError, match="File is empty"):
        validator.validate_file_bytes(b"", filename="empty.txt")


@pytest.mark.asyncio
async def test_download_and_validate_url_invalid_scheme():
    validator = IngestionSecurityValidator()
    with pytest.raises(SecurityValidationError, match="Unsupported URL scheme"):
        await validator.download_and_validate_url("ftp://example.com/file.pdf")


@pytest.mark.asyncio
async def test_download_and_validate_url_private_ip():
    validator = IngestionSecurityValidator()
    with pytest.raises(SecurityValidationError, match="prohibited IP address|Forbidden target IP"):
        await validator.download_and_validate_url("http://127.0.0.1:8000/document.pdf")
