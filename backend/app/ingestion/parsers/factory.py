"""
Parser Factory for MIME type dispatching (Section 6.5).
"""

from backend.app.ingestion.parsers.base import BaseParser
from backend.app.ingestion.parsers.pdf_parser import PDFParser
from backend.app.ingestion.parsers.docx_parser import DOCXParser
from backend.app.ingestion.parsers.html_parser import HTMLParser
from backend.app.ingestion.parsers.md_parser import MarkdownParser
from backend.app.ingestion.parsers.csv_parser import CSVParser


def get_parser_for_mime(mime_type: str) -> BaseParser:
    """Return appropriate parser for given validated MIME type."""
    if mime_type == "application/pdf":
        return PDFParser()
    elif mime_type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"):
        return DOCXParser()
    elif mime_type == "text/html":
        return HTMLParser()
    elif mime_type == "text/csv":
        return CSVParser()
    elif mime_type in ("text/markdown", "text/plain"):
        return MarkdownParser()
    else:
        # Default fallback to Markdown/Text parser
        return MarkdownParser()
