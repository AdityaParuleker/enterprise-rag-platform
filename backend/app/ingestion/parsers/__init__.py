"""
Parsers package for multi-format document extraction.
"""

from backend.app.ingestion.parsers.base import BaseParser, ParsedBlock
from backend.app.ingestion.parsers.factory import get_parser_for_mime

__all__ = ["BaseParser", "ParsedBlock", "get_parser_for_mime"]
